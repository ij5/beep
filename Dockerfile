FROM python:3

RUN pip install flask transformers

COPY new.py /

CMD ["python", "new.py"]