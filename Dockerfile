
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
#Insert an example inference_arguments.txt in the data repo to be bind mounted
COPY inference_arguments.txt /workspace/data
#Install other required packages, 
RUN python -m pip install -r /ctlesionseg/requirements.txt
#set default workdir with scripts and args
WORKDIR /workspace

#Q4: Does the program find everything like this?
#run the program, maye write as cmd
#RUN python run.py inference_wml_arguments.txt

#Q5: What does CMD do? exec param1 param2
#CMD ["python", "./run.py", "inference_arguments.txt"]
#ENTRYPOINT ["python ./run.py", "inference_arguments.txt"]

