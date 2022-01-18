#!/bin/bash
wget https://andrewowens.com/camo/camo-data.zip
unzip camo-data.zip -d ../
mv ../results/camo-data ../camo-data
python get_num_views.py
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gZJgqx4Lwp--oHAJQ3ZCwpWZHQEjncvO' -O animal_shapes.zip
unzip animal_shapes.zip -d ../
