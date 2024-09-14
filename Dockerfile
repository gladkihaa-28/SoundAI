FROM ubuntu:latest
MAINTAINER Alexander Gladkikh 'gaa.280811@gmail.com'
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app 
RUN pip3 install numpy==1.26.4 numba==0.60.0 librosa joblib imbalanced-learn scikit-learn
ENTRYPOINT ['python']
CMD ['main.py']
