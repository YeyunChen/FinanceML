import datetime as dt
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import data_util as dutil
from exercise import chap_02

__rlt_dir = "../plots/chap_19/"


def tick_rule(trades):
    trades['d_price'] = trades['price'].diff()
    trades['i_side'] = ''
    trades.loc[trades['d_price'] > 0, 'i_side'] = 'buy'
    trades.loc[trades['d_price'] < 0, 'i_side'] = 'sell'
    return trades


def evaluate_tick_rule(trades):
    trades = tick_rule(trades)

    # accuracy
    accuracy = len(trades.loc[trades['i_side'] == trades['side']]) / len(trades)
    print(f'accuracy of the tick rule is {accuracy:.0%}')

    # order book is thick, many trades executed at the same price, causing empty labels for the tick rule
    trades_0 = trades.loc[trades['d_price'] != 0]
    accuracy_0 = len(trades_0.loc[trades_0['i_side'] == trades_0['side']]) / len(trades_0)
    print(f'when there is a price move, accuracy of the tick rule is {accuracy_0:.0%}')

    # explore the relationship between volumes, price move, and the accuracy of tick rule
    trades['tick_rule'] = ''
    trades.loc[(trades['i_side'] != '') & (trades['i_side'] == trades['side']), 'tick_rule'] = 'correct'
    trades.loc[(trades['i_side'] != '') & (trades['i_side'] != trades['side']), 'tick_rule'] = 'incorrect'

    # sampling base on time bar
    freq = "5T"
    trades_a = trades.loc[trades['tick_rule'] == 'incorrect'].resample(freq).agg(
        {'price': np.count_nonzero}).rename({'price': 'n_incorrect'}, axis=1)
    trades_b = trades.resample(freq).agg({'side': np.count_nonzero, 'price': np.min, 'amount': np.sum}).rename(
        {'side': 'n_tick', 'price': 'price_l'}, axis=1)
    trades_c = trades.resample("5T").agg({'price': np.max}).rename(
        {'price': 'price_h'}, axis=1)
    trades_s = pd.concat([trades_a, trades_b, trades_c], axis=1)
    trades_s['incorrect_perc'] = trades_s['n_incorrect'] / trades_s['n_tick']
    trades_s['price_range'] = trades_s['price_h'] - trades_s['price_l']
    data_plot = trades_s[['price_range', 'amount', 'n_tick', 'n_incorrect', 'incorrect_perc']]
    sns.pairplot(data_plot)
    plt.savefig(__rlt_dir + "/tick_rule_pair_plot.")

    print("""
    the disagreement happens more often when there are high market activities, and when there are larger price jumps.
    """)

    return


def roll_model(trades, bar_size):
    print(f"form tick bar size = {bar_size}")
    tick_bar = chap_02.form_tick_bar(trades, bar_size=bar_size)
    tick_bar['log_price'] = np.log(tick_bar['price'])
    tick_bar['return'] = tick_bar['log_price'].diff()

    sigma = tick_bar['return'].std()
    serial_corr = tick_bar['return'].autocorr()
    serial_cov = serial_corr * sigma * sigma

    c = 0 if serial_cov > 0 else math.sqrt(0 - serial_cov)
    true_noice = sigma * sigma + 2 * serial_cov
    print(f"estimated value of true noise (sigma_u squared) is {true_noice}")
    print(f"estimated half spread is {c}")
    return c


def evaluate_roll_model(trades):
    source_date = dt.datetime(2020, 7, 17)
    trades_day = trades.loc[trades.index.date == dt.datetime(2020, 7, 17).date()].copy()

    tick_frequency = len(trades_day) / (24 * 60)
    print(f"avg tick per min is {tick_frequency}")

    h_spread_1 = roll_model(trades_day, bar_size=1)
    h_spread_5 = roll_model(trades_day, bar_size=5)
    h_spread_10 = roll_model(trades_day, bar_size=10)
    h_spread_15 = roll_model(trades_day, bar_size=15)

    quotes_tob = calc_tob_spread(source_date)
    quotes_tob_bar = quotes_tob.resample("10T").agg({'half_spread_bps': np.mean})

    fix, ax = plt.subplots(figsize=(10, 5))
    ax.plot(quotes_tob_bar.index, quotes_tob_bar['half_spread_bps'])
    ax.plot([quotes_tob_bar.index.min(), quotes_tob_bar.index.max()], [h_spread_1, h_spread_1],
            label='RollModelSpread_1')
    ax.plot([quotes_tob_bar.index.min(), quotes_tob_bar.index.max()], [h_spread_5, h_spread_5],
            label='RollModelSpread_5')
    ax.plot([quotes_tob_bar.index.min(), quotes_tob_bar.index.max()], [h_spread_10, h_spread_10],
            label='RollModelSpread_10')
    ax.plot([quotes_tob_bar.index.min(), quotes_tob_bar.index.max()], [h_spread_15, h_spread_15],
            label='RollModelSpread_15')

    plt.legend()
    plt.savefig(__rlt_dir + "/roll_model_spread_vs_real_spread")
    print(f'how the sample frequency affect the spread estimation?')
    print(f'the asset class seems illiquid (avg one trade per min)')
    print(f'roll model seems to be a good estimate of the bid/offer spread')
    print(f'think the sample frequenct should reflect the liquidity')
    print(f' - avg time it takes to get out of the quoting size')
    print(f' - to compute avg tob order size')
    print(f' - calc how long it takes to get out')


def calc_tob_spread(source_date):
    quotes = dutil.read_tob(source_date)
    quotes['half_spread'] = 0.5 * (quotes['tob_ask'] - quotes['tob_bid'])
    quotes['half_spread_bps'] = (quotes['tob_ask'] - quotes['tob_bid']) / (quotes['tob_ask'] + quotes['tob_bid'])
    return quotes


# def plot_factor_scatter(data, factors):
#     n = len(factors)
#     fix, ax = plt.subplots(n, n, figsize=(20, 15))
#
#     for i in range(n):
#         for j in range(n):
#             if i >= j:
#                 continue
#             ax[i, j].scatter(data[factors[i]], data[factors[j]])
#             plt.xlabel = factors[i]
#             plt.ylabel = factors[j]
#     plt.show()

# sns.scatterplot(trades_s, x='price_range', y='n_incorrect')
# plt.show()
# plot_factor_scatter(trades_s, ['price_range', 'amount', 'n_tick', 'n_incorrect', 'incorrect_perc'])

# fix, ax = plt.subplots(3)
# ax[0].scatter(trades_s['price_range'], trades_s['n_incorrect'])
# ax[1].scatter(trades_s['price_range'], trades_s['n_tick'])
# ax[2].scatter(trades_s['price_range'], trades_s['incorrect_perc'])
# plt.show()

# fix, ax = plt.subplots()
# ax.plot(trades_s.index, trades_s['n_incorrect'])
# ax2 = ax.twinx()
# ax2.plot(trades_s.index, trades_s['price_range'], color='orange')
# plt.show()

# plt.show()

if __name__ == "__main__":
    trades = dutil.read_trades()
    # evaluate_tick_rule(trades)
    # roll_model(trades)
    evaluate_roll_model(trades)
