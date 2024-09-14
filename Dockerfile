FROM ubuntu:latest
MAINTAINER Alexander Gladkikh 'gaa.280811@gmail.com'
RUN apt-get update -y
RUN sudo apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app 
RUN pip install -r requirements.txt
ENTRYPOINT ['python']
CMD ['main.py']
