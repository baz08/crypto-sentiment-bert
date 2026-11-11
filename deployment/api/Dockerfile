FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/api/requirements.txt

# CPU-only torch first (the default PyPI wheel bundles CUDA and is ~2.5 GB).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /code/api/requirements.txt

COPY ./ /code/api

ENV PYTHONPATH=/code/api
# Hugging Face Hub repo to load the fine-tuned model from.
ENV MODEL_REPO=baz08/crypto-sentiment-bert

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
