FROM python:3

RUN pip install flask transformers torch tensorflow

COPY new.py /

CMD ["python", "/new.py"]