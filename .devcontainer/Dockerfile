#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
#mcr.microsoft.com/devcontainers/python:0-3.8
FROM  nvidia/cuda:11.4.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y \
    curl \
    git \
    python3.8 \
    python3-pip
RUN pip3 install --upgrade pip
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install -U openmim && \
    mim install mmcv-full && \
    mim install mmdet && \
    mim install mmengine

RUN git clone https://github.com/open-mmlab/mmrotate.git && \
    pip install -e /mmrotate 

RUN mkdir -p ./src/contents && \
    mim download mmrotate --config redet_re50_refpn_1x_dota_ms_rr_le90 --dest ./src/contents

RUN pip install awscli