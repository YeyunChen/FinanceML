from data import data_util as dutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def tick_rule(trades):
    trades['d_price'] = trades['price'].diff()
    trades['i_side'] = ''
    trades.loc[trades['d_price'] > 0, 'i_side'] = 'buy'
    trades.loc[trades['d_price'] < 0, 'i_side'] = 'sell'

    accuracy = len(trades.loc[trades['i_side'] == trades['side']]) / len(trades)
    print(f'accuracy of the tick rule is {accuracy:.0%}')

    trades_0 = trades.loc[trades['d_price'] != 0]
    accuracy_0 = len(trades_0.loc[trades_0['i_side'] == trades_0['side']]) / len(trades_0)
    print(f'when there is a price move, accuracy of the tick rule is {accuracy_0:.0%}')

    trades['tick_rule'] = ''
    trades.loc[(trades['i_side'] != '') & (trades['i_side'] == trades['side']), 'tick_rule'] = 'correct'
    trades.loc[(trades['i_side'] != '') & (trades['i_side'] != trades['side']), 'tick_rule'] = 'incorrect'

    trades_a = trades.loc[trades['tick_rule'] == 'incorrect'].resample("5T").agg(
        {'price': np.count_nonzero}).rename({'price': 'n_incorrect'}, axis=1)
    trades_b = trades.resample("5T").agg({'side': np.count_nonzero, 'price': np.min, 'amount': np.sum}).rename(
        {'side': 'n_tick', 'price': 'price_l'}, axis=1)
    trades_c = trades.resample("5T").agg({'price': np.max}).rename(
        {'price': 'price_h'}, axis=1)
    trades_s = pd.concat([trades_a, trades_b, trades_c], axis=1)
    trades_s['incorrect_perc'] = trades_s['n_incorrect'] / trades_s['n_tick']
    trades_s['price_range'] = trades_s['price_h'] - trades_s['price_l']

    data_plot = trades_s[['price_range', 'amount', 'n_tick', 'n_incorrect', 'incorrect_perc']]
    sns.pairplot(data_plot)
    plt.savefig("tick_rule_pair_plot.jpeg")

    '''
    the disagreement happens more often when there are high market activities, and when there are larger price jumps.
    '''
    return trades


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
    tick_rule(trades)
