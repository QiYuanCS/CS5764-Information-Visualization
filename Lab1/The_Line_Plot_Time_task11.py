import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
url = ('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])
print(df.head())
Sales = df['Sales'].values
AdBudget = df['AdBudget'].values
GDP = df['GDP'].values
Time = df['Date'].values

plt.figure(figsize= (16,8))
plt.plot(Time,Sales, label='Sales', marker = 's')
plt.plot(Time,AdBudget, label='AdBudget', marker = 'o')
plt.plot(Time,GDP, label='GDP', marker = 'v')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sales, AdBudget and GDP Versus Time')

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gcf().autofmt_xdate(rotation = 45)
plt.legend()
plt.show()
