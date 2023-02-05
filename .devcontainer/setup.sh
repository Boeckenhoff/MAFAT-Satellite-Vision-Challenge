#!/bin/bash

mkdir -p ~/.cache/pydrive2fs/$GDRIVE_CLIENT_ID/
echo $DVC_GOOGLE_DRIVE > ~/.cache/pydrive2fs/$GDRIVE_CLIENT_ID/default.json

pip3 install --user -r .devcontainer/requirements.txt

mim install mmcv-full
mim install mmdet

git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -e .
