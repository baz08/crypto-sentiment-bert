from pmaw import PushshiftAPI
import pandas as pd
import json

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


crypto_df.to_csv('./BTC_comments.csv', header=True, index=False)
eth_df.to_csv('./ETH_comments.csv', header=True, index=False)
