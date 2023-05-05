import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


def getTEvents(gRaw, h):
    tEvents_b, tEvents_s, sPos, sNeg, ssPos, ssNeg = [], [], 0, 0, [0], [0]
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents_b.append(i)
        elif sPos > h:
            sPos = 0
            tEvents_s.append(i)
        ssPos.append(sPos)
        ssNeg.append(sNeg)
    return tEvents_b, tEvents_s, ssPos, ssNeg
    # return pd.DatetimeIndex(tEvents)



def uni_test():
    size = 1000
    returns = np.random.normal(0, 1, size)
    df = pd.DataFrame({'returns': returns}).reset_index()
    df['price'] = df['returns'].cumsum()

    events_b, events_s, p_pos, p_neg = getTEvents(df['price'], 10)
    df.loc[df.index.isin(events_b), 'event_b'] = df['price']
    df.loc[df.index.isin(events_s), 'event_s'] = df['price']
    df['p_pos'] = p_pos
    df['p_neg'] = p_neg

    sn.lineplot(df, x='index', y='price')
    sn.scatterplot(df, x='index', y='event_b', color='green')
    sn.scatterplot(df, x='index', y='event_s', color='red')

    ax2 = plt.twinx()
    sn.lineplot(df, x='index', y='p_pos', color='green', ax=ax2, alpha=0.15)
    sn.lineplot(df, x='index', y='p_neg', color='red', ax=ax2, alpha=0.15)
    plt.show()


if __name__ == '__main__':
    uni_test()
