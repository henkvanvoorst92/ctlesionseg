
###select right container image
#With STEP you control the use of cache or not: docker build --build-arg -t ctlesionseg .
#Pull base image with cuda capabilities 
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /
#Some base processing -- optional: 
RUN apt-get -y update && apt-get -y install git && pip install --upgrade pip && mkdir -p /workspace/data
#Fetch all files with scripts for the image, ST
COPY /files /ctlesionseg/files
COPY /requirements.txt /ctlesionseg/requirements.txt
COPY /inference_arguments.txt /workspace/inference_arguments.txt
#Alternative: RUN git clone https://github.com/henkvanvoorst92/ctlesionseg 
#Install other required packages
RUN python -m pip install -r /ctlesionseg/requirements.txt
#copy all models from the local pc to the image
COPY /models /models
#set default workdir with scripts and args
WORKDIR /workspace

