import pandas as pd
from cleaner import *


a_df = pd.read_csv("./BTC_comments.csv")
b_df = pd.read_csv("./ETH_comments.csv")
c_df = pd.read_csv("./sentiment.csv")
d_df = pd.read_csv("./sentiment_labels.csv")
c_df['body'] = c_df['Comment Text']


d_df['body'] = d_df['text']
d_df['Sentiment'] = d_df['sentiment']

d_df = d_df.drop(columns=['id', 'text', 'sentiment'])

c_df = c_df.drop(columns=['Comment Text', 'URL'])

df = pd.concat([a_df, b_df, c_df, d_df])

df['body'] = normalize_corpus(df['body'])

df['body'] = df['body'].replace(
    r'https\S+', '', regex=True).replace(r'www\S+', '', regex=True)

# DATA_COLUMN/LABEL_COLUMN matches what berttest.py expects. Subsetting to
# just these two also drops BTC/ETH_comments.csv's `author` column, so the
# merged training data doesn't carry Reddit usernames.
df = df.rename(columns={'body': 'DATA_COLUMN', 'Sentiment': 'LABEL_COLUMN'})
df = df[['DATA_COLUMN', 'LABEL_COLUMN']]

df.to_csv("./Crypto_com.csv", header=True, index=False)
