from data import data_generator as dataGen
import pandas as pd
import numpy as np


def load_data():
    data = dataGen.create_tick_data(n_samples=100000)
    return data


def form_tick_bar(data: pd.DataFrame, bar_size: int = 100):
    data['tick_count'] = range(len(data))
    data['tick_count_resi'] = data['tick_count'] % bar_size
    tick_bar = data.loc[data['tick_count_resi'] == 0, ['price']]
    return tick_bar


def form_volume_bar(data: pd.DataFrame, bar_size: int = 10000):
    data['volume_count'] = data['size'].cumsum()
    data['volume_multi'] = np.floor(data['volume_count'] / bar_size)
    data['volume_multi_chg'] = data['volume_multi'].diff()
    volume_bar = data.loc[data['volume_multi_chg'] != 0, ['price']]
    return volume_bar


def form_dollar_bar(data: pd.DataFrame, bar_size: int = 10000000):
    data['dollar_size'] = data['size'] * data['price']
    data['dollar_count'] = data['dollar_size'].cumsum()
    data['dollar_multi'] = np.floor(data['dollar_count'] / bar_size)
    data['dollar_multi_chg'] = data['dollar_multi'].diff()
    dollar_bar = data.loc[data['dollar_multi_chg'] != 0, ['price']]
    return dollar_bar


def roll_futures():
    return 1


if __name__ == '__main__':
    data = load_data()
    tick_bar = form_tick_bar(data)
    volume_bar = form_volume_bar(data)
    dollar_bar = form_dollar_bar(data)
