FROM mcr.microsoft.com/devcontainers/python:0-3.8

RUN git clone https://github.com/open-mmlab/mmrotate.git && \
    pip install -e /mmrotate

RUN pip3 install -U openmim && \
    mim install mmcv-full && \
    mim install mmdet


ARG PERSONAL_ACCESS_TOKEN
ENV DEBIAN_FRONTEND noninteractive
# enables using torchdata with s3
ENV S3_VERIFY_SSL '0'


# Install git (needed to clone lxUtils package)
RUN apt-get update && apt-get -y install git python3 python3-dev libz-dev python3-pip python3-opencv; apt-get clean

# Install cv2 dependencies (needed to clone lxClasses package)
RUN apt-get update && apt-get -y install libglib2.0-0 libsm6 ffmpeg libxexe6 libxrender1; apt-get clean


# Install requirements
COPY docker/docker_reqs.txt /docker_reqs.txt
RUN python3 -m pip install --upgrade pip --trusted-host pypi.python.org -r /docker_reqs.txt

ADD docker/train_docker_entrypoint.py /

WORKDIR /

