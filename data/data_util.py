import datetime as dt

import pandas as pd

__data_dir = "~/Documents/Projects/Data/practise"
__fn_quote = __data_dir + "/binance_iotabtc_orderbooks/{date_str}.csv"
__fn_trade = __data_dir + "/trades.csv"


def read_trades():
    df = pd.read_csv(__fn_trade)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index("timestamp")
    df['price'] = df['price'] * 1e+4
    return df


def read_quotes(source_date):
    fn = __fn_quote.format(date_str=source_date.strftime("%Y-%m-%d"))
    df = pd.read_csv(fn)
    return df


if __name__ == '__main__':
    trades = read_trades()
    print(f'trades file loaded {len(trades)} rows.')
    source_date = dt.datetime(2020, 7, 17)
    quotes = read_quotes(source_date)
    print(f'trades file loaded {len(quotes)} rows.')
