FROM pytorch/pytorch:latest

# System configuration and dependencies.
RUN apt-get update && apt-get upgrade -y && apt-get install software-properties-common -y

# RUN python -m ensurepip --upgrade
RUN pip install -U pip setuptools wheel Cython
RUN pip install pytorchvideo av torchvision tensorboard --upgrade

RUN add-apt-repository ppa:twodopeshaggy/drive && apt-get update && apt-get install drive -y

# ENTRYPOINT ["python", "main.py"]
