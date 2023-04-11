# Docker container: ctlesionseg
This repository contains the scripts used to construct a docker container for white matter lesion and follow-up infarct lesion segmentation in CT. The docker container can be found under henkvanvoorst92/ctlesionseg. All rights are reserved and no warrantees for model performance are given in for any use. To run this container a Linux OS is preferred, a GPU is required.

## How to install docker on your workstation
### For Linux OS:
https://docs.docker.com/engine/install/ubuntu/

### For Windows OS install windows subsystem for linux 2: 
Install WSL-2 for windows: https://learn.microsoft.com/nl-nl/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package
Ensure that you enable VM platform and HV: https://aka.ms/wsl2-install
Install docker for windows using wsl: https://docs.docker.com/desktop/windows/wsl/

## Prepare your data: Input folder
All CT scans should be in one folder stored as nifti images (extionsions: .nii.gz or .nii). The filename before the extension can be an ID, this ID will return in all the output files. In the same folder a inference_arguments.txt should be located so the docker knows what type of segmentations to return. In the example file in this github repo you can inspect the possible arguments and alter them if required. This folder should be used as the input folder to run the docker container. The output folder can be chosen.

## Run the docker container
To run the docker container the "Input folder" described above should be mounted to the container. Furthermore, the container should run in interactive mode.
An example of the command:

```
function test() {
  console.log("This code will have a copy button to the right of it");
}
```

## The output folders

