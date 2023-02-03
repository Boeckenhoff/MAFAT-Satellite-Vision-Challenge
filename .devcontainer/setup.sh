#!/bin/bash

pip3 install --user -r .devcontainer/requirements.txt

mim install mmcv-full
mim install mmdet

git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -e .
