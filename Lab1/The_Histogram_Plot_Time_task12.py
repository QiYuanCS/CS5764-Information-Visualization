import pandas as pd
import matplotlib.pyplot as plt

url = ('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
df = pd.read_csv(url)

Sales = df['Sales'].values
AdBudget = df['AdBudget'].values
GDP = df['GDP'].values
Time = df.index

plt.figure(figsize= (10,6))
plt.hist(Sales, bins=20, alpha=0.5, label='Values Sales', color='red')
plt.hist(AdBudget, bins=20, alpha=0.5, label='Values AdBudget', color='blue')
plt.hist(GDP, bins=20, alpha=0.5, label='Values GDP', color='green')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('The Histogram Plot of Sales, AdBudget and GDP')
plt.legend()
plt.show()
