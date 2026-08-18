FROM python:3.11

WORKDIR /code

COPY ./ /code/api

RUN pip install --upgrade pip \
    && pip install -r /code/api/requirements.txt

ENV PYTHONPATH=/code/api


EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]