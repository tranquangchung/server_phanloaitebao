FROM tensorflow/tensorflow:1.15.4-gpu-py3   

RUN apt-get update
COPY . /app
RUN pip install -r /app/requirements.txt
EXPOSE 8998
WORKDIR /app
CMD python app.py
