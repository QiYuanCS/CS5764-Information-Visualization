import pandas as pd

from Pearson_Correlation_Coefficient_task2 import Pearson_Correlation_Coefficient

url = ('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
df = pd.read_csv(url)

Sales = df['Sales'].values
AdBudget = df['AdBudget'].values
GDP = df['GDP'].values

r_Sales_AdBudget = Pearson_Correlation_Coefficient(Sales, AdBudget)
r_Sales_GDP = Pearson_Correlation_Coefficient(Sales, GDP)
r_AdBudget_GDP = Pearson_Correlation_Coefficient(AdBudget, GDP)

print(f'The sample Pearson’s correlation coefficient between Sales & AdBudget is: {r_Sales_AdBudget:.2f}')
print(f'The sample Pearson’s correlation coefficient between Sales & GDP is: {r_Sales_GDP:.2f}')
print(f'The sample Pearson’s correlation coefficient between AdBudget & GDP is: {r_AdBudget_GDP:.2f}')

