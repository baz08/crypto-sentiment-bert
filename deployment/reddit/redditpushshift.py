"""Scrape BTC / ETH comments from r/CryptoCurrency via the Pushshift API.

HISTORICAL ONLY: Reddit cut off public Pushshift access in May 2023, so this
script no longer returns data. It is kept to document how the comments in
BTC_comments.csv / ETH_comments.csv were collected (comments since
2022-08-09, bot account CointestMod excluded). Treat those CSVs as a frozen
snapshot.

The scraped comments had no labels; the Sentiment column in
BTC_comments.csv / ETH_comments.csv was added afterwards. This script
therefore writes to *_raw.csv so re-running it can never overwrite the
labeled files.
"""

from pmaw import PushshiftAPI
import pandas as pd

api = PushshiftAPI()

crypto = api.search_comments(q=('BTC', 'bitcoin'), subreddit="CryptoCurrency", filter=['ids', 'author', 'utc_datetime_str',
                                                                                       'body', 'subreddit'], limit=1000, since=1660037192)
crypto_list = [comment for comment in crypto]

eth = api.search_comments(q=('ETH', 'ethereum'), subreddit="CryptoCurrency", filter=['ids', 'author', 'utc_datetime_str',
                                                                                     'body', 'subreddit'], limit=1000, since=1660037192)
ETH_list = [comment for comment in eth]


crypto_df = pd.DataFrame(crypto_list)
eth_df = pd.DataFrame(ETH_list)


crypto_df = crypto_df[crypto_df['author'].str.contains('CointestMod') == False]
eth_df = eth_df[eth_df['author'].str.contains('CointestMod') == False]


crypto_df.to_csv('./BTC_comments_raw.csv', header=True, index=False)
eth_df.to_csv('./ETH_comments_raw.csv', header=True, index=False)
