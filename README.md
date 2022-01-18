# GANmouflage: 3D Object Nondetection with Texture Fields
Rui Guo<sup>1</sup> Jasmine Collins<sup>2</sup> Oscar de Lima<sup>1</sup> Andrew Owens<sup>1</sup>

<sup>1</sup>University of Michigan <sup>2</sup>UC Berkeley

Teaser Image!!!!

This repository includes codes for the paper: **GANmouflage: 3D Object Nondetection with Texture Fields**. arXiv: 

## Environment Setup
We provide instructions for creating a conda environment for training and generating camouflaged textures. 
```
conda create -n camo_env -y python=3.7
conda activate camo_env
sh ./env.sh
```
## Dataset
-   Scene image data can be downloaded from [link](https://andrewowens.com/camo/camo-data.zip). [[Owens et al., 2014]](https://andrewowens.com/camo/) Download the data and unzip data into the folder outside the code repository. Make sure scene data is in `../camo-data/`
    Then run 
    ```
    python get_num_views.py
    ```

-   Animal shapes can be downloaded from LINK. Animal shapes are collected from [SMAL](https://smal.is.tue.mpg.de/). We normalize the size of animals and flipped y-axis to accomodate to our axis definition. Download the data and unzip data into the folder outside the code repository. Make sure animal shape data is in `../fake_animals_v4/`

Or directly run
```
sh ./prepare_data.sh
```

## Training
A sample training command is included in `train_ddp.sh`. 

Scene name can be specified through `--scene SCENE_NAME`.

If you want to run the method on animal shapes use `--animals` 

## Generating Textures
A sample generating command is included in `generate.sh`.
