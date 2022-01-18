#!/bin/bash
wget https://andrewowens.com/camo/camo-data.zip
unzip camo-data.zip -d ../
mv ../results/camo-data ../camo-data
python get_num_views.py
