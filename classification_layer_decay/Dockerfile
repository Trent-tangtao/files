FROM docker.dm-ai.cn/nvidia/pytorch:20.11-py3
# FROM docker.dm-ai.cn/algorithm-research/gbits:21.03.18-3-cu11.1
# FROM docker.dm-ai.cn/algorithm-research/autotrain:10.0-cudnn7-devel-centos7-torch14-gcc7-autotrain
# RUN pip install --no-cache-dir -U polyaxon-client[gcs]==0.5.5
# RUN pip install --no-cache-dir -U polyaxon-client==0.5.3
RUN pip install --no-cache-dir -U polyaxon-client==0.5.3 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pip install -U timm==0.3.2
RUN pip install -U einops==0.3.0
RUN pip install -U tensorboardX
RUN pip install -U munkres
# RUN pip install -U deepspeed==0.4.0
RUN apt-get update
RUN apt-get install rsync -y
eNV SHELL /bin/bash

# Use bash as default shell, rather than sh
WORKDIR /code

COPY build/. /code
