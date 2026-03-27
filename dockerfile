FROM ubuntu:22.04

WORKDIR /IUM

RUN apt update
RUN apt install -y git python3-pip
RUN python3 -m pip install uv
RUN rm -rf /var/lib/apt/lists/*

RUN git clone https://git.wmi.amu.edu.pl/s481810/IUM26.git .
RUN uv sync