
###select right container image
#Pull base image with cuda capabilities 
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /
#Some base processing -- optional: 
RUN  apt-get -y update && apt-get -y install git && pip install --upgrade pip && mkdir -p /workspace/data
#Fetch all files with scripts for the image
RUN git clone https://github.com/henkvanvoorst92/ctlesionseg
#copy all models from the local pc to the image
COPY /models /models
#Install other required packages, /ctlesionseg/
RUN python -m pip install -r /ctlesionseg/requirements.txt
#set default workdir with scripts and args
WORKDIR /workspace

