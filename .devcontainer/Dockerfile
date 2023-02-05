FROM mcr.microsoft.com/devcontainers/python:0-3.8

RUN git clone https://github.com/open-mmlab/mmrotate.git && \
    pip install -e /mmrotate && \
    rm -r mmrotate

RUN pip3 install -U openmim && \
    mim install mmcv-full && \
    mim install mmdet

RUN mkdir -p ~/.cache/pydrive2fs/$GDRIVE_CLIENT_ID/ && \
    echo $DVC_GOOGLE_DRIVE > ~/.cache/pydrive2fs/$GDRIVE_CLIENT_ID/default.json