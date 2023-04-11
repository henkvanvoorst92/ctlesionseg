
###select right container image
#Pull base image with cuda capabilities 
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /
RUN  apt-get -y update && apt-get -y install git
#copy files and models to local
RUN git clone https://github.com/henkvanvoorst92/ctlesionseg/ /
COPY /models /models
COPY /inference_arguments.txt /workspace/data
RUN python -m pip install -r /requirements.txt
#set default workdir with scripts and args
WORKDIR /workspace

#Q4: Does the program find everything like this?
#run the program, maye write as cmd
#RUN python run.py inference_wml_arguments.txt

#Q5: What does CMD do? exec param1 param2
#CMD ["python", "./run.py", "inference_arguments.txt"]
#ENTRYPOINT ["python ./run.py", "inference_arguments.txt"]

