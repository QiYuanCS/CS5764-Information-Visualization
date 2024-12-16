import pandas as pd
import matplotlib.pyplot as plt

from Pearson_Correlation_Coefficient_task2 import Pearson_Correlation_Coefficient

url = ('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
df = pd.read_csv(url)

Sales = df['Sales'].values
AdBudget = df['AdBudget'].values
r_Sales_AdBudget = Pearson_Correlation_Coefficient(Sales, AdBudget)
print(f'The sample Pearson’s correlation coefficient between Sales & AdBudget is: {r_Sales_AdBudget:.2f}')
plt.scatter(Sales, AdBudget, alpha = 0.5, color = 'blue', label= 'Sales & AdBudget')
plt.title(f'Scatter plot of AdBudget & Sales\nCorrelation Coefficient: {r_Sales_AdBudget:.2f}')
plt.xlabel('Variable Sales')
plt.ylabel('Variable AdBudget')
plt.grid()
plt.legend()
plt.show()