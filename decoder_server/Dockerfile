FROM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y pip

RUN pip install Flask --ignore-installed
RUN pip install einops

COPY . /home

EXPOSE 8123
