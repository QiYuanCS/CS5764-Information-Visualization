import pandas as pd
from pandas_datareader import data
import yfinance as yf
from prettytable import PrettyTable
yf.pdr_override()

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print('\n#1')
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = "2013-01-01"
end_date = "2024-05-22"
df = data.get_data_yahoo(stocks, start=start_date, end=end_date).round(2)
table = PrettyTable()
table.field_names = ['Date'] + df.columns.tolist()
for index, row in df.tail().iterrows():
    table.add_row([index.strftime("%Y-%m-%d")] + [f"{value:.2f}" for value in row.tolist()])

print(table)
for index, row in df.iterrows():
    table.add_row([index.strftime("%Y-%m-%d")] + [f"{value:.2f}" for value in row.tolist()])

#print(table)

print('\n#2 3 4 5')
attributes = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
dfs = {}
for stat in ['mean', 'var', 'std', 'median']:
    stat_data = {attr: df[attr].agg(stat).round(2) for attr in attributes}
    stat_df = pd.DataFrame(stat_data, index=stocks)

    stat_df.loc['Maximum Value'] = stat_df.max().round(2)
    stat_df.loc['Minimum Value'] = stat_df.min().round(2)
    max_company = stat_df.idxmax()
    min_company = stat_df.idxmin()
    stat_df.loc['Maximum company name'] = max_company
    stat_df.loc['Minimum company name'] = min_company

    table = PrettyTable()
    table.title = f"{stat.capitalize()} Value Comparison"
    table.field_names = ['Name \\ Features '] + [f"{col}($)" if col != 'Volume' else col for col in stat_df.columns]

    for row in stat_df.itertuples():
        format_row = [row.Index] + [f"{value:.2f}" if isinstance(value, (int, float))
                                    else value for value in row[1:]]
        table.add_row(format_row)
    dfs[stat] = table
    print(f'\n{stat.upper()} Table:')
    print(table)

print('\n#6')
def display_correlation(stock, start, end):
    df = data.get_data_yahoo(stock, start=start, end=end).round(2)
    correlation_matrix = df.corr().round(2)

    table = PrettyTable()
    table.title = f"{stock} Value Comparison"
    table.field_names = ['Attribute'] + df.columns.tolist()
    for row in correlation_matrix.itertuples():
        table.add_row([row.Index] + [f'{value:.2f}' for value in list(row[1:])])

    print(f'\n{stock.upper()} Correlation Matrix:')
    print(table)

display_correlation('AAPL', start_date, end_date)

print('\n#7')
for stock in stocks[1:]:
    display_correlation(stock, start_date, end_date)

#%%
print('\n#8')
adj_close = df['Adj Close']
returns = adj_close.pct_change().dropna()
std_dev = returns.std().round(3)
comparison_table = PrettyTable()
comparison_table.field_names = ['Symbol', 'Return Std Dev', 'Risk Level']

quantiles = std_dev.quantile([0.33, 0.66])
threshold_low = quantiles.iloc[0]
threshold_high = quantiles.iloc[1]

for stock in stocks:
    sigma = std_dev[stock]
    if sigma < threshold_low:
        risk = 'low'
    elif sigma < threshold_high:
        risk = 'medium'
    else:
        risk = 'high'
    comparison_table.add_row([stock, f"{sigma}", risk])

print('\n Stock Volatility Comparison:')
print(comparison_table)

safer_stock = std_dev.idxmin()
risker_stock = std_dev.idxmax()
print(f'\n Safer Investment: {safer_stock}')
print(f'\n Risker Investment: {risker_stock}')
