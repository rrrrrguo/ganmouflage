import kornia.augmentation.random_generator as rg
from typing import Tuple, List, Union, Dict, Optional, cast
import random
from kornia.augmentation import AugmentationBase2D
import torch
from torch.distributions import Bernoulli
import kornia.augmentation.functional as F
from kornia.constants import Resample, BorderType, SamplePadding
from kornia.geometry import bbox_generator
from kornia.augmentation.utils import (
    _adapted_uniform,
    _common_param_check,
)
from kornia.utils import _extract_device_dtype
import warnings
import torch.nn as nn


def get_bbox_from_mask(masks):
    '''
    Parameters
    ----------
    masks         [bs,H,W]

    Returns
    -------
        bbox    [bs,4] in (y1,x1,y2,x2) format
    '''
    bboxes=[]
    for mask in masks:
        _,y, x = torch.nonzero(mask, as_tuple=True)
        if len(y)==0:
            y=torch.tensor([masks.shape[1]//2])
            x=torch.tensor([masks.shape[2]//2])
        bboxes.append([y.min().item(), x.min().item(), y.max().item(),x.max().item()])
    return torch.tensor(bboxes)


def fixed_crop_generator(
        batch_size: int,
        input_size: Tuple[int, int],
        size: Union[Tuple[int, int], torch.Tensor],
        obj_bbox: torch.Tensor,
        resize_to: Optional[Tuple[int, int]] = None,
        same_on_batch: bool = False,
        offset=(-0.25,0.25),
        dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```crop``` transformation for crop transform.
    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (h, w).
        size (tuple): Desired size of the crop operation, like (h, w).
            If tensor, it must be (B, 2).
        obj_bbox (torch.Tensor): Place of the target, (B,4) (y1,x1,y2,x2)
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.
    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (torch.Tensor): output bounding boxes with a shape (B, 4, 2).
    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    Example:
        >>> _ = torch.manual_seed(0)
        >>> crop_size = torch.tensor([[25, 28], [27, 29], [26, 28]])
        >>> random_crop_generator(3, (30, 30), size=crop_size, same_on_batch=False)
        {'src': tensor([[[ 1.,  0.],
                 [28.,  0.],
                 [28., 24.],
                 [ 1., 24.]],
        <BLANKLINE>
                [[ 1.,  1.],
                 [29.,  1.],
                 [29., 27.],
                 [ 1., 27.]],
        <BLANKLINE>
                [[ 0.,  3.],
                 [27.,  3.],
                 [27., 28.],
                 [ 0., 28.]]]), 'dst': tensor([[[ 0.,  0.],
                 [27.,  0.],
                 [27., 24.],
                 [ 0., 24.]],
        <BLANKLINE>
                [[ 0.,  0.],
                 [28.,  0.],
                 [28., 26.],
                 [ 0., 26.]],
        <BLANKLINE>
                [[ 0.,  0.],
                 [27.,  0.],
                 [27., 25.],
                 [ 0., 25.]]])}
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([size if isinstance(size, torch.Tensor) else None])
    # Use float point instead
    _dtype = _dtype if _dtype in [torch.float16, torch.float32, torch.float64] else dtype
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=_device, dtype=_dtype).repeat(batch_size, 1)
    else:
        size = size.to(device=_device, dtype=_dtype)
    assert size.shape == torch.Size([batch_size, 2]), (
        "If `size` is a tensor, it must be shaped as (B, 2). "
        f"Got {size.shape} while expecting {torch.Size([batch_size, 2])}.")
    assert input_size[0] > 0 and input_size[1] > 0 and (size > 0).all(), \
        f"Got non-positive input size or size. {input_size}, {size}."
    size = size.floor()

    x_diff = input_size[1] - size[:, 1] + 1
    y_diff = input_size[0] - size[:, 0] + 1

    # Start point will be 0 if diff < 0
    x_diff = x_diff.clamp(0)
    y_diff = y_diff.clamp(0)

    if batch_size == 0:
        return dict(
            src=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            dst=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
        )
    center = (obj_bbox[:, :2] + obj_bbox[:, 2:]) / 2
    yx_start_lower = center - size * (0.5-offset[0])
    yx_start_upper = center - size * (0.5-offset[1])  # [B,2]

    y_start_lower = yx_start_lower[:, 0]
    x_start_lower = yx_start_lower[:, 1]
    y_start_upper = yx_start_upper[:, 0]
    x_start_upper = yx_start_upper[:, 1]

    y_start_lower=y_start_lower.clamp(min=0)
    x_start_lower=x_start_lower.clamp(min=0)
    y_start_upper=torch.minimum(y_start_upper,y_diff)
    x_start_upper=torch.minimum(x_start_upper,x_diff)

    x_start = _adapted_uniform((1,), x_start_lower, x_start_upper, same_on_batch).floor()
    y_start = _adapted_uniform((1,), y_start_lower, y_start_upper, same_on_batch).floor()

    crop_src = bbox_generator(
        x_start.view(-1).to(device=_device, dtype=_dtype),
        y_start.view(-1).to(device=_device, dtype=_dtype),
        torch.where(size[:, 1] == 0, torch.tensor(input_size[1], device=_device, dtype=_dtype), size[:, 1]),
        torch.where(size[:, 0] == 0, torch.tensor(input_size[0], device=_device, dtype=_dtype), size[:, 0]))

    if resize_to is None:
        crop_dst = bbox_generator(
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            size[:, 1],
            size[:, 0])
    else:
        assert len(resize_to) == 2 and isinstance(resize_to[0], (int,)) and isinstance(resize_to[1], (int,)) \
               and resize_to[0] > 0 and resize_to[1] > 0, \
            f"`resize_to` must be a tuple of 2 positive integers. Got {resize_to}."
        crop_dst = torch.tensor([[
            [0, 0],
            [resize_to[1] - 1, 0],
            [resize_to[1] - 1, resize_to[0] - 1],
            [0, resize_to[0] - 1],
        ]], device=_device, dtype=_dtype).repeat(batch_size, 1, 1)

    return dict(src=crop_src,
                dst=crop_dst)


class RandomResizedCropAroundTarget(nn.Module):
    def __init__(
            self, size: Tuple[int, int], scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
            ratio: Union[torch.Tensor, Tuple[float, float]] = (3. / 4., 4. / 3.),
            offset=(-0.25,0.25),
            resample: Union[str, int, Resample] = Resample.BILINEAR.name,
            same_on_batch: bool = False,
            align_corners: bool = True

    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens all the time.
        super(RandomResizedCropAroundTarget, self).__init__()
        self._device, self._dtype = _extract_device_dtype([scale, ratio])
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.resample: Resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )
        self.same_on_batch = same_on_batch
        self.offset=offset

    def __repr__(self) -> str:
        repr = f"size={self.size}, scale={self.scale}, ratio={self.ratio}, interpolation={self.resample.name}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def forward(self,
                in_tensor: torch.Tensor,
                obj_bbox: torch.Tensor,  # [bs,4] bbox [x1,y1,x2,y2]
                ) -> torch.Tensor:  # type: ignore

        batch_shape = in_tensor.shape
        params = self.generate_parameters(batch_shape, obj_bbox)
        self._params = params
        output = self.apply_transform(in_tensor, self._params)
        return output

    def generate_parameters(self, batch_shape: torch.Size, obj_bbox: torch.Tensor) -> Dict[str, torch.Tensor]:
        scale = torch.as_tensor(self.scale, device=self._device, dtype=self._dtype)
        ratio = torch.as_tensor(self.ratio, device=self._device, dtype=self._dtype)
        target_size: torch.Tensor = rg.random_crop_size_generator(
            batch_shape[0], self.size, scale, ratio, self.same_on_batch, self._device, self._dtype)['size']
        return fixed_crop_generator(batch_shape[0], (batch_shape[-2], batch_shape[-1]), target_size,
                                    obj_bbox,
                                    resize_to=self.size, same_on_batch=self.same_on_batch,
                                    offset=self.offset, dtype=self._dtype)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_crop(input, params, self.flags)
