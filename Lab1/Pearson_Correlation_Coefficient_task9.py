import pandas as pd
import matplotlib.pyplot as plt

from Pearson_Correlation_Coefficient_task2 import Pearson_Correlation_Coefficient

url = ('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
df = pd.read_csv(url)

Sales = df['Sales'].values
GDP = df['GDP'].values

r_Sales_GDP = Pearson_Correlation_Coefficient(Sales, GDP)

print(f'The sample Pearson’s correlation coefficient between Sales & GDP is: {r_Sales_GDP:.2f}')

plt.scatter(Sales, GDP, alpha = 0.5, color = 'blue', label= 'Sales & GDP')
plt.title(f'Scatter plot of GDP & Sales\nCorrelation Coefficient: {r_Sales_GDP:.2f}')

plt.xlabel('Variable Sales')
plt.ylabel('Variable GDP')
plt.grid()
plt.legend()
plt.show()