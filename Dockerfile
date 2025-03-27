# Dockerfile for FAICORD  
# write by Jaedong, Oh (2025.03.26)
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE}
WORKDIR /speech-recognition 
COPY . .
ENV LC_ALL=ko_KR.UTF-8 

RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8   
RUN apt-get install python3-pip -y
RUN apt-get install vim -y 
RUN apt-get update && apt-get install git -y
RUN pip install -r requirements.txt
