FROM ubuntu:latest
MAINTAINER Alexander Gladkikh 'gaa.280811@gmail.com'
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app 
RUN pip install -r requirements.txt
ENTRYPOINT ['python']
CMD ['main.py']
