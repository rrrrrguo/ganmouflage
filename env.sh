
conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -c conda-forge
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
pip install open3d pandas kornia==0.5.0 pandas opencv-python trimesh[easy] matplotlib sklearn scikit-image tensorboardX tqdm lpips
conda install -y -c conda-forge pyembree
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0