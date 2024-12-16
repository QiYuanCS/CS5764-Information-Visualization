import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
print('#1')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

url = 'https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/raw/refs/heads/main/Sample%20-%20Superstore.xls'
df = pd.read_excel(url)
df_droped = df.drop(['Row ID', 'Order ID', 'Customer ID','Customer Name', 'Postal Code', 'Product ID','Order Date', 'Ship Date', 'Country', 'Segment'],
                    axis = 1)
print(df_droped.head())

#%%
print('#2')
df_category = df_droped.groupby('Category').sum()

print(df_category)

#%%
max_profit_df_category = df_category['Profit'].idxmax()
min_profit_df_category = df_category['Profit'].idxmin()
max_discount_df_category = df_category['Discount'].idxmax()
min_discount_df_category = df_category['Discount'].idxmin()
max_quantity_df_category = df_category['Quantity'].idxmax()
min_quantity_df_category = df_category['Quantity'].idxmin()
max_sales_df_category = df_category['Sales'].idxmax()
min_sales_df_category = df_category['Sales'].idxmin()

print(f'Max Profit Category: {max_profit_df_category} Max Profit:{max(df_category['Profit']):.2f}'
      f' Min Profit Category: {min_profit_df_category} Min Profit:{min(df_category['Profit']):.2f}')
print(f'Max Discount Category: {max_discount_df_category} Max Discount:{max(df_category['Discount']):.2f}'
      f' Min Discount Category: {min_discount_df_category} Min Discount:{min(df_category['Discount']):.2f}')
print(f'Max Quantity Category: {max_quantity_df_category} Max Quantity:{max(df_category['Quantity']):.2f}'
      f' Min Quantity Category: {min_quantity_df_category} Min Quantity:{min(df_category['Quantity']):.2f}')
print(f'Max Sales Category: {max_sales_df_category} Max Sales:{max(df_category['Sales']):.2f}'
      f' Min Sales Category: {min_sales_df_category} Min Sales:{min(df_category['Sales']):.2f}')
#%%
fig, axes = plt.subplots(2, 2, figsize = (18, 18))
explode_profit = [0.1 if category == min_profit_df_category else 0 for category in df_category.index]
axes[0, 0].pie(df_category['Profit'], labels = df_category.index, autopct='%1.2f%%',
               startangle=90, explode=explode_profit, textprops={'fontsize': 30})
axes[0, 0].set_title('Total Profit by Category', fontdict={'family':'serif', 'color':'blue', 'size':35})

explode_profit = [0.1 if category == min_discount_df_category else 0 for category in df_category.index]
axes[0, 1].pie(df_category['Discount'], labels = df_category.index, autopct='%1.2f%%',
               startangle=90, explode=explode_profit, textprops={'fontsize': 30})
axes[0, 1].set_title('Total Discount', fontdict={'family':'serif', 'color':'blue', 'size':35})

explode_profit = [0.1 if category == min_quantity_df_category else 0 for category in df_category.index]
axes[1, 0].pie(df_category['Quantity'], labels = df_category.index, autopct='%1.2f%%',
               startangle=90, explode=explode_profit, textprops={'fontsize': 30})
axes[1, 0].set_title('Total Quantity', fontdict={'family':'serif', 'color':'blue', 'size':35})

explode_profit = [0.1 if category == min_sales_df_category else 0 for category in df_category.index]
axes[1, 1].pie(df_category['Sales'], labels = df_category.index, autopct='%1.2f%%',
               startangle=90, explode=explode_profit, textprops={'fontsize': 30})
axes[1, 1].set_title('Total Sales', fontdict={'family':'serif', 'color':'blue', 'size':35})
plt.tight_layout()
plt.show()
#%%
print('#3')
from prettytable import PrettyTable
table = PrettyTable()
table.title = "Super store - Category"
table.field_names = ["\\","Sales($)", "Quantity", "Discounts($)", "Profit($)"]

categories = ['Furniture', 'Office Supplies', 'Technology']

for category in categories:
    if category in df_category.index:
        category_data = df_category.loc[category]
        table.add_row([category, f'{category_data['Sales']:.2f}', f'{category_data['Quantity']:.2f}',
                       f'{category_data['Discount']:.2f}', f'{category_data['Profit']:.2f}'])

max_sales = df_category['Sales'].max()
min_sales = df_category['Sales'].min()
max_quantity = df_category['Quantity'].max()
min_quantity = df_category['Quantity'].min()
max_discount = df_category['Discount'].max()
min_discount = df_category['Discount'].min()
max_profit = df_category['Profit'].max()
min_profit = df_category['Profit'].min()

table.add_row(["Maximum Value", f'{max_sales:.2f}', f'{max_quantity:.2f}',f'{max_discount:.2f}',f'{max_profit:.2f}'])
table.add_row(["Minmum Value", f'{min_sales:.2f}', f'{min_quantity:.2f}', f'{min_discount:.2f}',f'{min_profit:.2f}'])

max_sales_df_category = df_category['Sales'].idxmax()
min_sales_df_category = df_category['Sales'].idxmin()
max_quantity_df_category = df_category['Quantity'].idxmax()
min_quantity_df_category = df_category['Quantity'].idxmin()
max_discount_df_category = df_category['Discount'].idxmax()
min_discount_df_category = df_category['Discount'].idxmin()
max_profit_df_category = df_category['Profit'].idxmax()
min_profit_df_category = df_category['Profit'].idxmin()

table.add_row(["Maximum Feature", max_sales_df_category, max_quantity_df_category, max_discount_df_category, max_profit_df_category])
table.add_row(["Minimum Feature", min_sales_df_category, min_quantity_df_category, min_discount_df_category, min_profit_df_category])

print(table)

#%%
print('#4')
df_sub_category_sales = df.groupby('Sub-Category')['Sales'].sum()
df_sub_category_profit = df.groupby('Sub-Category')['Profit'].sum()
sub_categories = ['Phones', 'Chairs', 'Storage', 'Tables', 'Binders', 'Machines', 'Accessories', 'Copiers', 'Bookcases', 'Appliances']
fig, ax1 = plt.subplots(figsize=(20, 8))
bar_width = 0.4
bars = ax1.bar(sub_categories, df_sub_category_sales[sub_categories], color='#95DEE3', edgecolor='blue', width=bar_width, label='Sales', zorder=3)
for bar in bars:
    if bar.get_height() > 250000:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height()*0.7, f'${bar.get_height():.2f}',
                 horizontalalignment='center', va='bottom', fontsize=20, rotation=90)
    else:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'${bar.get_height():.2f}',
                 horizontalalignment='center', va='bottom', fontsize=20, rotation=90)
ax1.set_ylabel('USD($)', fontsize=25)
y_min = min(df_sub_category_profit.min(), df_sub_category_sales.min()) - 50000
y_max = max(df_sub_category_profit.max(), df_sub_category_sales.max()) + 50000
ax1.set_ylim(y_min, y_max)
ax2 = ax1.twinx()
ax2.plot(sub_categories, df_sub_category_profit[sub_categories], color='red', linewidth=4, marker='o', label='Profit')
ax2.set_ylabel('USD($)', fontsize=25)
ax2.set_ylim(y_min, y_max)
ax1.set_title('Sales and Profit per sub-category', fontsize=30)
ax1.set_xlabel('Sales', fontsize=25)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax1.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 1))
ax2.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 0.9))
fig.patch.set_facecolor('white')
ax1.grid(True, zorder=0)
plt.tight_layout()
plt.show()

#%%
print('#5')
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
font1 = {'family':'serif', 'color':'blue', 'size':20}
font2 = {'family': 'serif', 'color':'darkred', 'size':15}

plt.plot(x,y1, '--', label='sine wave', color='blue', lw=3)
plt.plot(x,y2,'-.', label='cosine wave', color='red', lw=3)
plt.fill_between(x, y1, y2, where=(y1 > y2), alpha=0.3, color = 'green')
plt.fill_between(x, y1, y2, where=(y1 < y2), alpha=0.3, color='orange')
plt.annotate('area where sine is greater than cosine',
             fontsize=10,
             xy=(2, 0.25), xytext=(3, 1),
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green'))
plt.title('Fill between x-axis and plot line', font1)
plt.xlabel('x-axis', font2)
plt.ylabel('y-axis', font2)
plt.tight_layout()
plt.grid()
plt.legend(loc='lower left', fontsize=15, prop={'weight': 'bold'})
plt.show()


#%%
print('#6')
x = np.arange(-4, 4, 0.01)
y = np.arange(-4, 4, 0.01)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y ** 2))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=1, edgecolor='none', alpha=0.7)
ax.contour(X, Y, Z, zdir='z', offset=-6, cmap='coolwarm', linewidths=1)
ax.contour(X, Y, Z, zdir='x', offset=-5, cmap='coolwarm', linewidths=1)
ax.contour(X, Y, Z, zdir='y', offset=5, cmap='coolwarm', linewidths=1)
ax.set_title(r'surface plot of $z = \sin \sqrt{x^2+y^2}$', fontdict={'family':'serif', 'color':'blue', 'size':25})
ax.set_xlim([-4.5, 4.5])
ax.set_ylim([-4.5, 4.5])
ax.set_zlim([-6, 2])
ax.set_xlabel('X label', fontdict={'family':'serif', 'color':'darkred', 'size':15})
ax.set_ylabel('Y label', fontdict={'family':'serif', 'color':'darkred', 'size':15})
ax.set_zlabel('Z label', fontdict={'family':'serif', 'color':'darkred', 'size':15})
ax.set_xticks(np.arange(-4, 5, 1))
ax.set_yticks(np.arange(-4, 5, 1))
ax.set_zticks(np.arange(-6, 3, 1))
plt.tight_layout()
plt.show()
#%%
print('#7')
df_sub_category_sales = df.groupby('Sub-Category')['Sales'].sum()
df_sub_category_profit = df.groupby('Sub-Category')['Profit'].sum()
sub_categories = ['Phones', 'Chairs', 'Storage', 'Tables', 'Binders', 'Machines', 'Accessories', 'Copiers', 'Bookcases', 'Appliances']
bar_width = 0.4
x = np.arange(len(sub_categories))*1.65
fig = plt.figure(figsize=(9, 7))

ax1 = fig.add_subplot(2, 1, 1)
ax1.bar(x - bar_width / 2, df_sub_category_sales.loc[sub_categories], color='#95DEE3', edgecolor='blue', width=bar_width,
        label='Sales', tick_label=sub_categories)
ax1.bar(x + bar_width / 2, df_sub_category_profit.loc[sub_categories], color='lightcoral', edgecolor='red', width=bar_width,
        label='Profit', tick_label=sub_categories)
ax1.set_title('Sales and Profit per sub-category', fontsize=15)
ax1.set_xlabel('Sales', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(sub_categories, fontsize=10)

ax1.set_ylabel('USD($)', fontsize=10)
ax1.set_ylim(bottom=-50000)
ax1.grid()
ax1.legend(loc='upper right')
df_sales_category = df.groupby('Category')['Sales'].sum()
df_profit_category = df.groupby('Category')['Profit'].sum()
ax2 = fig.add_subplot(2, 2, 3)
ax2.pie(df_sales_category, labels=df_sales_category.index, autopct='%1.2f%%', startangle=0)
ax2.set_title('Sales', fontsize=15)
ax3 = fig.add_subplot(2, 2, 4)
ax3.pie(df_profit_category, labels=df_profit_category.index, autopct='%1.2f%%', startangle=0)
ax3.set_title('Profit', fontsize=15)
plt.tight_layout()
plt.show()



