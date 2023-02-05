FROM mcr.microsoft.com/devcontainers/python:0-3.8

RUN git clone https://github.com/open-mmlab/mmrotate.git && \
    pip install -e /mmrotate && \
    rm -r mmrotate

RUN pip3 install -U openmim && \
    mim install mmcv-full && \
    mim install mmdet

RUN pip install awcsli && \
    echo 'aws_access_key_id={$S3_ACESS_TOKEN$}' >> ~/.aws/credentials && \
    echo 'aws_secret_access_key={$S3_SECRET_KEY$}' >> ~/.aws/credentials