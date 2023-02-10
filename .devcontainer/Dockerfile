FROM mcr.microsoft.com/devcontainers/python:0-3.8

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/open-mmlab/mmcv.git  
ENV MMCV_WITH_OPS=1 
RUN python3 -m pip install -e mmcv

#RUN pip3 install -U openmim && \
#    mim install mmcv-full && \
#    mim install mmdet

RUN git clone https://github.com/open-mmlab/mmrotate.git && \
    pip install -e /mmrotate 
   # rm -r /mmrotate
