FROM pytorch/pytorch:latest

RUN apt update && apt install -y \
vim \
libx11-6 \
imagemagick \ 
 && rm -rf /var/lib/apt/lists/* 

# Jupyter Notebook config
COPY docker/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 9998

#RUN conda config --add channels https://repo.continuum.io/pkgs/main

WORKDIR "/root/projects"
COPY . /root/projects/pytorch_fnet
WORKDIR "/root/projects/pytorch_fnet"

#RUN conda env create --name pytorch_fnet -f environment.yml

ENV PATH /opt/conda/envs/pytorch_fnet/bin:$PATH
#ENV CONDA_DEFAULT_ENV pytorch_fnet
#ENV CONDA_PREFIX /opt/conda/envs/pytorch_fnet

RUN pip install -e . 
