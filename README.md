# Docker container: ctlesionseg
This repository contains the scripts used to construct a docker image and container for white matter lesion and follow-up infarct lesion segmentation in CT. The docker container can be found under henkvanvoorst92/ctlesionseg. All rights are reserved and no warrantees for model performance are given in for any use. To run this container a Linux OS is preferred, a GPU is required.\
Docker hub link:  https://hub.docker.com/repository/docker/henkvanvoorst92/ctlesionseg/general

## How to install docker on your workstation
### For Linux OS:
https://docs.docker.com/engine/install/ubuntu/

### For Windows OS install windows subsystem for linux 2: 
Install WSL-2 for windows: https://learn.microsoft.com/nl-nl/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package
\
Ensure that you enable VM platform and HV: https://aka.ms/wsl2-install
\
Install docker for windows using wsl: https://docs.docker.com/desktop/windows/wsl/

## Prepare your data: Input folder
All CT scans should be in one folder stored as nifti images (extionsions: .nii.gz or .nii). The filename before the extension can be an ID, this ID will return in all the output files. In the same folder a inference_arguments.txt should be located so the docker knows what type of segmentations to return. In the example file in this github repo you can inspect the possible arguments and alter them if required. This folder should be used as the input folder to run the docker container. The output folder can be chosen.

## Run the docker container
To run the docker container the "Input folder" described above should be mounted (-v) to the worspace/data folder in the container (ctlesionseg). Furthermore, the container should run in interactive mode (-ti). 
\
Read more on using bind mount of a local folder to a docker container :  https://docs.docker.com/storage/bind-mounts/
\
An example of the command for windows:

```
docker run -ti -v C:\usr\documents\yourfolder:/workspace/data ctlesionseg
```

Now the container starts in interactive mode. In the command line you can execute model inference by calling with python the run.py file. The run.py file requires an inference_arguments.txt file where all the choice for input parameters are defined as tuples. If you use the same folder for input (/workspace/data) as for output, all output folders and files will be returned to your local disk.

Below an example that should execute white matter lesion segmentation and return results in a mounted folder. For this example we mounted a local C:\usr\documents\data folder. A sub folder of our local folder (\originals) contains CT files (ID.nii.gz). In the C:\usr\documents\data the above stated inference_arguments.txt file is also inserted (please inspect the arguments used carefully).

```
python /ctlesionseg/files/run.py /workspace/data/inference_arguments.txt
```

## The output folders
In the output folder several intermediate processed folders are available for inspection and the final output folder. 

Intermediate processing folders:\
/brainmask: Masks of the brain that are used to remove background, eroded versions are used to reduce the ventricle mask.\
/imagesTs: The CTs processed as input for the nnUnet models for inference.\
/predicted: Output files of the nnUnet when the imagesTs are used as input.\
/ventriclemask: For white matter lesion segmentation a ventricle mask is used to select periventricular (by default in a 10mm radius) lesions.\
\
Final results folder:\
/results: Contains all the relevant files per ID in a subfolder.
