FROM pytorch/pytorch:latest

ADD get-pip.py .

ADD requirements.txt .

RUN python get-pip.py
RUN pip install -r requirements.txt