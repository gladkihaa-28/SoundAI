FROM ubuntu:latest
MAINTAINER Alexander Gladkikh 'gaa.280811@gmail.com'
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app 
RUN apt install python3-numpy python3-numba python3-librosa python3-joblib python3-imbalanced-learn python3-scikit-learn
ENTRYPOINT ['python']
CMD ['main.py']
