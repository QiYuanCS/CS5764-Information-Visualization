import pandas as pd
import numpy as np
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override()

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
#%%
stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

start_date = "2000-01-01"
end_date = "2023-08-28"
df = data.get_data_yahoo(stocks, start=start_date, end=end_date).round(2)
#%%
print('\n#1')
highs = df['High'][stocks]
axes = highs.plot(
    subplots=True,
    layout=(3, 2),
    figsize=(16, 8),
    linewidth=3,
    fontsize=15,
    grid=True,
    color='C0',
    sharex=False
)

for ax, stock in zip(axes.flatten(), stocks):
    ax.clear()
    highs[stock].dropna().plot(
        ax=ax,
        linewidth=3,
        fontsize=15,
        grid=True,
        color='C0',
        label=stock
    )
    ax.set_title(f'High Price history of {stock}', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('High Price USD($)', fontsize=15)
    ax.legend([stock], loc='upper left', fontsize=15)
    ax.set_xlim([highs[stock].dropna().index.min(), highs[stock].dropna().index.max()])
    ax.tick_params(axis='x', labelrotation=0)

plt.tight_layout()
plt.show()

#%%
print('\n#2')
new_features = ['Low', 'Open', 'Close', 'Volume', 'Adj Close']

for feature in new_features:
    data = df[feature][stocks]

    axes = data.plot(
        subplots=True,
        layout=(3, 2),
        figsize=(16, 8),
        linewidth=3,
        fontsize=15,
        grid=True,
        color='C0',
        sharex=False
    )

    for ax, stock in zip(axes.flatten(), data.columns):
        if feature != 'Volume':
            ax.set_title(f'{feature} price history of {stock}', fontsize=15)
            ax.set_ylabel(f'{feature} USD($)', fontsize=15)
            ax.legend([stock], loc='upper left', fontsize=15)
        else:
            ax.set_title(f'Volume history of {stock}', fontsize=15)

            ax.set_ylabel('Volume', fontsize=15)
            ax.legend([stock], loc='upper right', fontsize=15)

        ax.set_xlim([df[feature][stock].dropna().index.min(), df[feature][stock].dropna().index.max()])
        ax.set_xlabel('Date', fontsize=15)
        ax.tick_params(axis='x', labelrotation=0)

    plt.tight_layout()
    plt.show()




#%%
print('\n#3')
highs = df['High'][stocks]
axes = highs.plot(
    kind='hist',
    subplots=True,
    layout=(3, 2),
    figsize=(16, 8),
    alpha=0.9,
    grid=True,
    fontsize=15,
    sharex=False
)

for ax, stock in zip(axes.flatten(), highs.columns):
    stock_min = highs[stock].min()
    stock_max = highs[stock].max()
    q75, q25 = np.percentile(highs[stock].dropna(), [75, 25])
    iqr = q75 - q25
    bins = min(max(int((stock_max - stock_min) / (2 * iqr)), 40), 90)
    ax.clear()

    highs[stock].plot(
        kind='hist',
        bins=bins,
        ax=ax,
        alpha=0.9,
        grid=True
    )
    ax.set_title(f'High Price history of {stock}', fontsize=15)
    ax.set_xlabel('Value in USD($)', fontsize=15)
    ax.set_ylabel('Frequency', fontsize=15)
    ax.legend([stock], loc='upper right', fontsize=15)
    ax.grid(True)

plt.tight_layout()
plt.show()



#%%
print('\n#4')
new_features = ['Low', 'Open', 'Close', 'Volume', 'Adj Close']
for feature in new_features:
    data = df[feature][stocks]
    axes = data.plot(
        kind = 'hist',
        subplots=True,
        layout=(3, 2),
        figsize=(16, 8),
        alpha=0.9,
        grid=True,
        fontsize=15,
        sharex=False
    )
    axes = axes.flatten()

    for ax,stock in zip(axes, stocks):
        stock_data = data[stock]
        stock_min = stock_data.min()
        stock_max = stock_data.max()

        q75, q25 = np.percentile(stock_data.dropna(), [75, 25])
        iqr = q75 - q25
        bins = min(max(int((stock_max - stock_min) / (2 * iqr)), 40), 90)
        ax.clear()
        stock_data.plot(
            kind='hist',
            bins=bins,
            ax=ax,
            alpha=0.9,
            grid=True
        )
        if feature != 'Volume':
            ax.set_title(f'{feature} price history of {stock}', fontsize = 15)
            ax.set_ylabel('Frequency', fontsize=15)
            ax.set_xlabel('Value in USD($)', fontsize=15)
        else:
            ax.set_title(f'Volume history of {stock}', fontsize = 15)
            ax.set_xlabel(f'Volume', fontsize=15)
            ax.set_ylabel('Frequency', fontsize=15)
            ax.legend(loc='upper right', fontsize=15)

        ax.legend(loc='upper right', fontsize=15)

        ax.grid(True)

    plt.tight_layout()
    plt.show()

#%%
import pandas as pd
import numpy as np
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override()

print('\n#5')
stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

start_date = "2000-01-01"
end_date = "2023-08-28"

Apple_data = data.get_data_yahoo(stocks[0], start = start_date, end = end_date).round(2)
scatter_matrix = pd.plotting.scatter_matrix(Apple_data[features],
                                            alpha = 0.5,
                                            figsize = (15, 15),
                                            diagonal = 'kde',
                                            hist_kwds = {'bins': 50},
                                            s = 10)
plt.tight_layout()
plt.show()

#%%
print('\n#6')
import pandas as pd
import numpy as np
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override()

features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

start_date = "2000-01-01"
end_date = "2023-08-28"
remain_stocks = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']

for stock in remain_stocks:
    stock_data = data.get_data_yahoo(stock, start = start_date, end = end_date).round(2)
    stock_scatter_matrix = pd.plotting.scatter_matrix(stock_data[features],
                                                alpha=0.5,
                                                figsize=(15, 15),
                                                diagonal='kde',
                                                hist_kwds={'bins': 50},
                                                s=10)
    plt.tight_layout()
    plt.show()

