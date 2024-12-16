import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

#%%
print('1')
penguins = sns.load_dataset('penguins')
print(penguins.tail())
print(penguins.describe().round(2))

#%%
print('2')
missing_isna = penguins.isna().sum()
missing_isnull= penguins.isnull().sum()
print("Missing values is na: ")
print(missing_isna)
print("\b Missing values is null: ")
print(missing_isnull)

penguins_cleaned = penguins.dropna()

cleaned_isna = penguins_cleaned.isna().sum()
cleaned_isnull = penguins_cleaned.isnull().sum()
print("\nMissing values after cleaning is na:")
print(cleaned_isna)
print("\nMissing values after cleaning is null:")
print(cleaned_isnull)

#%%
print('3')
sns.set_theme(style='darkgrid')
plt.figure(figsize=(8, 6))
sns.histplot(penguins_cleaned['flipper_length_mm'], kde = True)
plt.title('Histogram of Flipper Length with KDE')
plt.xlabel('Flipper Length mm')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('4')
sns.histplot(penguins_cleaned['flipper_length_mm'], kde = True, binwidth=3)
plt.title('Histogram of Flipper Length with KDE binwidth=3')
plt.xlabel('Flipper Length mm')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('5')
plt.figure(figsize=(8, 6))
sns.histplot(penguins_cleaned['flipper_length_mm'], kde=True, bins=30)

plt.title('Histogram of Flipper Length with KDE Bins = 30')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('6')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='species', kind='hist', kde=True)
plt.title('Histogram of Flipper Length per Species')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('7')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='species', kind='hist', kde=True, element='step')
plt.title('Histogram of Flipper Length per Species (Step)')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('8')
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins_cleaned, x='flipper_length_mm', hue='species', multiple='stack', kde=True)
plt.title('Histogram of Flipper Length per Species (Step)')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('9')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='sex', multiple='dodge', kind='hist')
plt.title('Histogram of Flipper Length per Species (Dodge)')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('10')
plt.figure(figsize=(8, 6))
g = sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='sex', kind='hist', col='sex', kde=True)
g.fig.suptitle('Separate Histograms of Flipper Length for Male and Female Penguins', y=1.05)
g.set_axis_labels('Flipper Length (mm)', 'Flipper Count')
plt.tight_layout()
plt.show()

#%%
print('11')
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins_cleaned, x='flipper_length_mm', hue='species', stat='density', common_norm=True, kde=True)
plt.title('Normalized Histogram of Flipper Length by Species')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('12')
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins_cleaned, x='flipper_length_mm', hue='sex', stat='density', common_norm=True, kde=True)
plt.title('Normalized Histogram of Flipper Length by Sex')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('13')
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins_cleaned, x='flipper_length_mm', hue='species', stat='probability', common_norm=True, kde=True)
plt.title('Normalized Histogram of Flipper Length by Species (Probability)')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Probability')
plt.tight_layout()
plt.show()

#%%
print('14')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='species', kind='kde')
plt.title('KDE of Flipper Length by Species')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('15')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='sex', kind='kde')

plt.title('KDE of Flipper Length by Sex')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('16')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='species', kind='kde', multiple='stack')

plt.title('Stacked KDE of Flipper Length by Species')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('17')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='sex', kind='kde', multiple='stack')

plt.title('Stacked KDE of Flipper Length by Sex')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('18')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='species', kind='kde', fill=True)
plt.title('KDE of Flipper Length by Species with Fill')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

#%%
print('19')
plt.figure(figsize=(8, 6))
sns.displot(data=penguins_cleaned, x='flipper_length_mm', hue='sex', kind='kde', fill=True)
plt.title('KDE of Flipper Length by Sex with Fill')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()


#%%
print('20')
plt.figure(figsize=(8, 6))
sns.lmplot(data=penguins_cleaned, x='bill_length_mm', y='bill_depth_mm')

plt.title('Regression Line of Bill Length vs Bill Depth')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.tight_layout()
plt.show()

#%%
print('21')
plt.figure(figsize=(8, 6))
sns.countplot(data=penguins_cleaned, x='island', hue='species')
plt.title('Count Plot of Penguins by Island and Species')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.tight_layout()
plt.show()

#%%
print('22')
plt.figure(figsize=(8, 6))
sns.countplot(data=penguins_cleaned, x='sex', hue='species')

plt.title('Count Plot of Penguins by Sex and Species')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#%%
print('23')
plt.figure(figsize=(8, 6))
sns.kdeplot(data=penguins_cleaned, x='bill_length_mm', y='bill_depth_mm', hue='sex', fill=True)
plt.title('Bivariate KDE of Bill Length vs Bill Depth by Sex')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.tight_layout()
plt.show()

#%%
print('24')
plt.figure(figsize=(8, 6))
sns.kdeplot(data= penguins_cleaned, x='bill_length_mm', y='flipper_length_mm', hue='sex', fill=True)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the penguins dataset
penguins = sns.load_dataset('penguins')

# Define figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot for Question 24: Bivariate KDE of Bill Length vs Flipper Length by Sex
sns.kdeplot(data=penguins, x='bill_length_mm', y='flipper_length_mm', hue='sex', fill=True, ax=axes[0])
axes[0].set_title('Bill Length vs Flipper Length by Sex')
axes[0].set_xlabel('Bill Length (mm)')
axes[0].set_ylabel('Flipper Length (mm)')

# Plot for Question 25: Bivariate KDE of Flipper Length vs Bill Depth by Sex
sns.kdeplot(data=penguins, x='flipper_length_mm', y='bill_depth_mm', hue='sex', fill=True, ax=axes[1])
axes[1].set_title('Flipper Length vs Bill Depth by Sex')
axes[1].set_xlabel('Flipper Length (mm)')
axes[1].set_ylabel('Bill Depth (mm)')

# Plot for Question 23: Bivariate KDE of Bill Length vs Bill Depth by Sex
sns.kdeplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='sex', fill=True, ax=axes[2])
axes[2].set_title('Bill Length vs Bill Depth by Sex')
axes[2].set_xlabel('Bill Length (mm)')
axes[2].set_ylabel('Bill Depth (mm)')

# Adjust layout for clarity
plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the penguins dataset
penguins = sns.load_dataset('penguins')

# Define figure with 3 rows for bivariate KDE plots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Plot for Question 27: Bivariate KDE of Bill Length vs Bill Depth by Sex
sns.kdeplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='sex', fill=True, ax=axes[0])
axes[0].set_title('Bill Length vs Bill Depth by Sex')
axes[0].set_xlabel('Bill Length (mm)')
axes[0].set_ylabel('Bill Depth (mm)')

# Plot for Question 28: Bivariate KDE of Bill Length vs Flipper Length by Sex
sns.kdeplot(data=penguins, x='bill_length_mm', y='flipper_length_mm', hue='sex', fill=True, ax=axes[1])
axes[1].set_title('Bill Length vs Flipper Length by Sex')
axes[1].set_xlabel('Bill Length (mm)')
axes[1].set_ylabel('Flipper Length (mm)')

# Plot for Question 29: Bivariate KDE of Flipper Length vs Bill Depth by Sex
sns.kdeplot(data=penguins, x='flipper_length_mm', y='bill_depth_mm', hue='sex', fill=True, ax=axes[2])
axes[2].set_title('Flipper Length vs Bill Depth by Sex')
axes[2].set_xlabel('Flipper Length (mm)')
axes[2].set_ylabel('Bill Depth (mm)')

# Adjust layout for clarity
plt.tight_layout()
plt.show()

#%%
sns.kdeplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='sex', fill=True)

# Add title and labels
plt.title('Bivariate KDE of Bill Length vs Bill Depth by Sex')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.tight_layout()
# Show the plot
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the penguins dataset
penguins = sns.load_dataset('penguins')

# Create a hexbin bivariate plot for 'bill_length_mm' vs 'bill_depth_mm' for male and female penguins
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='sex', bins=30, pthresh=.1, cmap="light:b", cbar=True)

# Add title and labels
plt.title('Question 27: Bill Length vs Bill Depth by Sex')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')

# Show the plot
plt.show()



#%%
# Re-load the penguins dataset again since it was reset earlier
penguins = sns.load_dataset('penguins')

# Recreate the block-style (binned) plot similar to the one user shared
plt.figure(figsize=(8, 6))

# Use histplot to generate a binned plot for 'bill_length_mm' vs 'bill_depth_mm' for male and female penguins
sns.histplot(data=penguins_cleaned, x='bill_length_mm', y='bill_depth_mm', hue='sex', bins=20, cbar=True, pthresh=0.1)

# Add title and labels
plt.title('Question 27: Bill Length vs Bill Depth by Sex (Binned)')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()

