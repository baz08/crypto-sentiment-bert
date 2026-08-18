# Sentiment Analysis of Cryptocurrency Reddit Comments using BERT

## Crypto Sentiment Analysis Deployment Guide 

This guide outlines the steps for deploying the Crypto Sentiment Analysis application.

### 1. Training
The model is already pretrained from 3000 datapoints on Reddit comments of cryptocurrencies. The BERT model can be found either at Huggingface at baz08/crypto-Bert-test or at the directory on Huggingface:
      https://huggingface.co/baz08/crypto-Bert-test          
Further training can be done via `deployment/Bert Training/berttest.py` (see `WRITEUP.md` in that folder for methodology and results).



### 2. Source Tree

├── docker
│   └──api - main.py
│       └── ML
│           ├──predmodel.py - BERT model loaded
│           └── __init__.py     
├── docs
│   └── UCSD MLE Capstone Project.pdf
├──Bert Training
│   ├──berttest.py - Script to train/evaluate the BERT model
│   └──WRITEUP.md - Training methodology and results write-up
└ reddit - Source of data from training


### 3. FastAPI server
Fast API server runs predictions on available devices and returns a sentiment. You can run the application through your terminal with the following:

```
docker run -p 8000:8000 deployment

The following endpoints are available:
|               URL           | Method |                 Description                 |
|-----------------------------|--------|---------------------------------------------|
|http://localhost:8000/predict| POST   | Run sentiment analysis on the provided text |

The api can be utilized and accessed through the link:  http://localhost:8000/docs

FastAPI has an Automatic Documentation built in, making it incredibly easy to make a request through the api. 
