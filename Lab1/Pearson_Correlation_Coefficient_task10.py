import pandas as pd
import matplotlib.pyplot as plt

from Pearson_Correlation_Coefficient_task2 import Pearson_Correlation_Coefficient

url = ('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv')
df = pd.read_csv(url)


GDP = df['GDP'].values
AdBudget = df['AdBudget'].values

r_GDP_AdBudget = Pearson_Correlation_Coefficient(GDP, AdBudget)

print(f'The sample Pearson’s correlation coefficient between GDP & AdBudget is: {r_GDP_AdBudget:.2f}')

plt.scatter(AdBudget, GDP, alpha = 0.5, color = 'blue', label= 'GDP & AdBudget')

plt.title(f'Scatter plot of GDP & AdBudget\nCorrelation Coefficient: {r_GDP_AdBudget:.2f}')

plt.xlabel('Variable AdBudget')
plt.ylabel('Variable GDP')
plt.grid()
plt.legend()
plt.show()