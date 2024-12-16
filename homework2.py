import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)

#%%
print('#1')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url)
df_cleaned = df.dropna()
print(df_cleaned)

#%%
print('#2')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url)
df_cleaned = df.dropna()
China_Columns = [col for col in df_cleaned.columns if 'China' in col]
df_cleaned.loc[:,China_Columns] = df_cleaned.loc[:, China_Columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['China_sum'] = df_cleaned[China_Columns].sum(axis=1)

China_columns = [col for col in df.columns if 'China' in col]
df['China_sum'] = df[China_columns].sum(axis=1)
print(df[['China.1','China.2','China.3', 'China.4', 'China.5', 'China.6', 'China_sum']])

#%%
print('#3')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url)
df_cleaned = df.dropna()
United_Kingdom_Columns = [col for col in df_cleaned.columns if 'United Kingdom' in col]
df_cleaned.loc[:,United_Kingdom_Columns] = df_cleaned.loc[:, United_Kingdom_Columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['United Kingdom_sum'] = df_cleaned[United_Kingdom_Columns].sum(axis=1)

UK_columns = [col for col in df.columns if 'United Kingdom' in col]
df_cleaned['United Kingdom_sum'] = df[UK_columns].sum(axis=1)

print(df[['United Kingdom.1','United Kingdom.2','United Kingdom.3', 'United Kingdom.4', 'United Kingdom.5', 'United Kingdom.6', 'United Kingdom_sum']])


#%%
print('#4')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url, index_col=0, skiprows=[1])
df = df[~df.index.isin(['Country/Region', 'Province/State'])]
df = df.dropna()
df.index = pd.to_datetime(df.index, format='%m/%d/%y')
us_data = df['US'].astype(float)
plt.figure(figsize=(12, 6))
plt.plot(us_data.index, us_data.values, label='US Confirmed Cases')
plt.xlabel('Year')
plt.ylabel('Confirmed COVID-19 Cases')
plt.title('US Confirmed COVID-19 Cases')
plt.legend(loc='upper left')
plt.grid()
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
feb_date = pd.Timestamp('2020-02-01')
ax.annotate('2020', xy=(feb_date, 0), xytext=(0, -45),
            textcoords='offset points', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

#%%
print('#5')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url, index_col=0, skiprows=[1])
df = df[~df.index.isin(['Country/Region', 'Province/State'])]
df = df.dropna()
df.index = pd.to_datetime(df.index, format='%m/%d/%y')

China_Columns = [col for col in df.columns if 'China' in col]
df['China_sum'] = df[China_Columns].sum(axis=1)
UK_columns = [col for col in df.columns if 'United Kingdom' in col]
df['United Kingdom_sum'] = df[UK_columns].sum(axis=1)

countries = ['United Kingdom_sum','China_sum','US','Germany','Brazil', 'India', 'Italy']
plt.figure(figsize=(12, 6))
for country in countries:
    plt.plot(df.index, df[country], label=country)

plt.xlabel('Year')
plt.ylabel('Confirmed COVID-19 cases')
plt.title('Global confirmed COVID19 cases')

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
feb_date = pd.Timestamp('2020-02-01')
ax.annotate('2020', xy=(feb_date, 0), xytext=(0, -45),
            textcoords='offset points', ha='center', fontsize=10)
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

#%%
print('#6')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url, index_col=0, skiprows=[1])
df = df[~df.index.isin(['Country/Region', 'Province/State'])]
df = df.dropna()
df.index = pd.to_datetime(df.index, format='%m/%d/%y')
us_data = df['US'].astype(float)

plt.figure(figsize=(12, 6))
plt.hist(us_data.index, bins=50, weights=us_data.values, label='US confirmed COVID-19 cases')
plt.xlabel('Date')
plt.ylabel('Confirmed COVID-19 cases')
plt.title('Histogram of US Confirmed COVID-19 Cases')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

#%%
print('#7')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url, index_col=0, skiprows=[1])
df = df[~df.index.isin(['Country/Region', 'Province/State'])]
df = df.dropna()
df.index = pd.to_datetime(df.index, format='%m/%d/%y')

China_Columns = [col for col in df.columns if 'China' in col]
df['China_sum'] = df[China_Columns].sum(axis=1)
UK_columns = [col for col in df.columns if 'United Kingdom' in col]
df['United Kingdom_sum'] = df[UK_columns].sum(axis=1)
countries = ['United Kingdom_sum','China_sum', 'Italy', 'Brazil', 'Germany', 'India']

fig, axes = plt.subplots(3,2, figsize=(16, 10))
axes = axes.flatten()

for i, country in enumerate(countries):
    ax = axes[i]
    ax.hist(df[country].index, bins=50, weights=df[country], label=f'{country} confirmed COVID-19 cases')
    ax.set_title(f'{country} confirmed COVID-19 cases')
    ax.set_xlabel('Date')
    ax.set_ylabel('Confirmed COVID-19 cases')
    ax.legend(loc='upper left')
    ax.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

#%%
print('#8')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url, index_col=0, skiprows=[1])
df = df[~df.index.isin(['Country/Region', 'Province/State'])]
df = df.dropna()
df.index = pd.to_datetime(df.index, format='%m/%d/%y')
China_Columns = [col for col in df.columns if 'China' in col]
df['China_sum'] = df[China_Columns].sum(axis=1)
UK_columns = [col for col in df.columns if 'United Kingdom' in col]
df['United Kingdom_sum'] = df[UK_columns].sum(axis=1)
countries = ['United Kingdom_sum','China_sum', 'US', 'Italy', 'Brazil', 'Germany', 'India']
means = {}
variances = {}
medians = {}
for country in countries:
    means[country] = df[country].mean()
    variances[country] = df[country].var()
    medians[country] = df[country].median()

calculate_df = pd.DataFrame({
    'country': countries,
    'mean': [means[country] for country in countries],
    'variance': [variances[country] for country in countries],
    'median': [medians[country] for country in countries]
})
calculate_df.set_index('country', inplace=True)
calculate_df = calculate_df.map(lambda x: f'{x:.2f}')
print(calculate_df)

highest_mean_country = calculate_df['mean'].astype(float).idxmax()
highest_variance_country = calculate_df['variance'].astype(float).idxmax()
highest_median_country = calculate_df['median'].astype(float).idxmax()
print(f'highest_mean_country:       {highest_mean_country}, mean = {means[highest_mean_country]:.2f}')
print(f'highest_variance_country:   {highest_variance_country}, variance = {variances[highest_variance_country]:.2f}')
print(f'highest_median_country:     {highest_median_country}, median = {medians[highest_median_country]:.2f}')

#%%
print('#Q7-1')
import seaborn as sns
titantic = sns.load_dataset('titanic')
missing_number = titantic.isna().sum()
missing_features = missing_number[missing_number > 0]

print('\nnan entries')
print(missing_number)
print('\nnan entries number')
print(missing_features)

titantic_cleaned = titantic.dropna()
missing_number_drop = titantic_cleaned.isna().sum()
missing_features_drop = missing_number_drop[missing_number_drop > 0]

print('\nRemove all the nan in the dataset')
print('\nnan entries ')
print(missing_number_drop)
print('\nnan entries number')
print(missing_features_drop)
if missing_features_drop.empty:
    print('\nThe dataset is cleaned')
    print(missing_features_drop)
    print('\nThe first five rows of the dataset')
    print(titantic_cleaned.head())

#%%
print('#Q7-2')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
gender_counts = titanic_cleaned['sex'].value_counts()

male_count = gender_counts.get('male', 0)
female_count = gender_counts.get('female', 0)

print(f"Total number of males: {male_count}")
print(f"Total number of females: {female_count}")

labels = ['Male', 'Female']
sizes = [male_count, female_count]


plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct=lambda p: f'{int(p * sum(sizes) / 100)}', startangle=90, colors = ['lightblue', 'lightcoral'])
plt.title('The total number of males and females ')
plt.axis('equal')
plt.legend(labels, title="Gender")
plt.show()

#%%
print('#Q7-3')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
gender_count = titanic_cleaned['sex'].value_counts()
gender_percentages = (gender_count / gender_count.sum()) * 100
print('Percentage of males and females:')
print(gender_percentages.round(2))

plt.figure(figsize = (8, 8))
plt.pie(gender_percentages, labels = gender_percentages.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
plt.title('The percentage of male and female on the titanic dataset')
plt.axis('equal')
plt.legend(labels, title="Gender")
plt.show()

#%%
print('#Q7-4')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
male_data = titanic_cleaned[titanic_cleaned['sex'] == 'male']
male_survival_count = male_data['survived'].value_counts()
print(f'Males Survived: {male_survival_count[1]}')
print(f'Males dit not Survive: {male_survival_count[0]}')

male_survival_percentages = (male_survival_count / male_survival_count.sum()) * 100
plt.figure(figsize=(8, 8))
plt.pie(male_survival_percentages, labels=['Did not survive', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
plt.title('The percentage of males who survived versus the percentage of males who did not survive')
plt.axis('equal')
plt.legend(labels, title="Gender")
plt.show()

#%%
print('#Q7-5')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
female_data = titanic_cleaned[titanic_cleaned['sex'] == 'female']
female_survival_count = male_data['survived'].value_counts()
print(f'Females Survived: {male_survival_count[1]}')
print(f'Females dit not Survive: {male_survival_count[0]}')

male_survival_percentages = (male_survival_count / male_survival_count.sum()) * 100
plt.figure(figsize=(10, 10))
plt.pie(male_survival_percentages, labels=['Did not survive', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
plt.title('The percentage of Females who survived versus the percentage of Females who did not survive')
plt.axis('equal')
plt.legend(labels, title="Gender")
plt.show()

#%%
print('#Q7-6')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
class_count = titanic_cleaned['pclass'].value_counts().sort_index()

print('The number of passengers in each class:')
print(f'First Class: {class_count[1]}')
print(f'Second Class: {class_count[2]}')
print(f'Third Class: {class_count[3]}')

class_percentages = (class_count / class_count.sum()) * 100
plt.figure(figsize=(10, 10))
plt.pie(class_percentages, labels=['First Class', 'Second Class', 'Third Class'], autopct='%1.1f%%', startangle=90,
        colors=['skyblue', 'lightpink','gold'])
plt.title('The percentage passengers with first class, second class and third-class tickets:')
plt.axis('equal')
plt.legend(labels, title="Gender")
plt.show()

#%%
print('#Q7-7')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
survival_count_class = titanic_cleaned.groupby('pclass')['survived'].sum()
count_class = titanic_cleaned['pclass'].value_counts().sort_index()

survival_class_percentages = (survival_count_class / count_class.sum()) * 100
print('The number of survivors in each class:')
print(survival_count_class)
print('The percentages of survivors in each class:')
print(survival_class_percentages.round(2))

plt.figure(figsize=(10, 10))
plt.pie(survival_class_percentages, labels=['First Class', 'Second Class', 'Third Class'], autopct='%1.1f%%', startangle=90,
        colors=['skyblue', 'lightpink','gold'])
plt.title('The percentage of survivors in each class:')
plt.axis('equal')
plt.legend(labels, title="Gender")
plt.show()

#%%
print('#Q7-8')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
def plot_survival_class(class_number):
    class_data = titanic_cleaned[titanic_cleaned['pclass'] == class_number]
    survival_count = class_data['survived'].value_counts()

    print(f'\nFirst Class: {class_number} passengers:')
    print(f'Survived: {survival_count[1]}')
    print(f'Did not survive: {survival_count[0]}')

    survival_percentages = (survival_count / survival_count.sum()) * 100
    plt.figure(figsize=(10, 10))
    plt.pie(survival_percentages, labels=['Did not survive', 'Survived'],autopct='%1.1f%%', startangle=90,
            colors=['skyblue', 'lightpink', 'gold']
            )
    plt.title(f'Survival Percentage of Class {class_number}')
    plt.axis('equal')
    plt.legend(labels, title="Gender")
    plt.show()

for class_number in range(1, 4):
    plot_survival_class(class_number)

#%%
print('#Q7-9')
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.dropna()
gender_count = titanic_cleaned['sex'].value_counts()
male_count = gender_counts.get('male', 0)
female_count = gender_counts.get('female', 0)
labels = ['Male', 'Female']
sizes = [male_count, female_count]
male_data = titanic_cleaned[titanic_cleaned['sex'] == 'male']
male_survival_count = male_data['survived'].value_counts()
male_survival_percentages = (male_survival_count / male_survival_count.sum()) * 100

female_data = titanic_cleaned[titanic_cleaned['sex'] == 'female']
female_survival_count = female_data['survived'].value_counts()
female_survival_percentages = (female_survival_count / female_survival_count.sum()) * 100

class_count = titanic_cleaned['pclass'].value_counts().sort_index()
class_percentages = (class_count / class_count.sum()) * 100

survival_count_class = titanic_cleaned.groupby('pclass')['survived'].sum()
count_class = titanic_cleaned['pclass'].value_counts().sort_index()
survival_class_percentages = (survival_count_class / count_class.sum()) * 100

class1 = titanic_cleaned[titanic_cleaned['pclass'] == 1]
class1_survival = class1['survived'].value_counts()
class1_survival_percentage = (class1_survival / class1_survival.sum()) * 100

class2 = titanic_cleaned[titanic_cleaned['pclass'] == 2]
class2_survival = class2['survived'].value_counts()
class2_survival_percentage = (class2_survival / class2_survival.sum()) * 100

class3 = titanic_cleaned[titanic_cleaned['pclass'] == 1]
class3_survival = class3['survived'].value_counts()
class3_survival_percentage = (class3_survival / class3_survival.sum()) * 100

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))
def plot_pie1(ax, labels, data, title, colors):
    ax.pie(data, labels=labels, autopct=lambda p: f'{int(p * sum(data) / 100)}', startangle=90, colors=colors)
    ax.set_title(title)
    ax.axis('equal')
def plot_pie(ax, labels, data, title, colors):
    ax.pie(data, labels = labels,  autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title(title)
    ax.axis('equal')

plot_pie1(axes[0, 0], labels, sizes, 'The number of Male and Female on Titanic', colors = ['skyblue', 'lightpink'])
plot_pie(axes[0, 1], gender_percentages.index, gender_percentages, 'The Percentage of Male and Female', ['skyblue', 'lightpink'])
plot_pie(axes[0, 2], ['Did not survive', 'Survived'], male_survival_percentages, 'The Percentage of Male Survival ', ['skyblue', 'lightpink'])

plot_pie(axes[1, 0], ['Did not survive', 'Survived'], female_survival_percentages, 'The Percentage of Female Survival ', ['skyblue', 'lightpink'])
plot_pie(axes[1, 1], ['First Class', 'Second Class', 'Third Class'], class_percentages, 'Passengers by Class', ['gold', 'skyblue', 'lightpink'])
plot_pie(axes[1, 2], ['First Class', 'Second Class', 'Third Class'], survival_class_percentages, 'Survivors by Class', ['gold', 'skyblue', 'lightpink'])

plot_pie(axes[2, 0], ['Did not survive', 'Survived'], class1_survival_percentage, 'Survival Percentage - First Class', ['skyblue', 'lightpink'])
plot_pie(axes[2, 1], ['Did not survive', 'Survived'], class2_survival_percentage, 'Survival Percentage - Second Class', ['skyblue', 'lightpink'])
plot_pie(axes[2, 2], ['Did not survive', 'Survived'], class3_survival_percentage, 'Survival Percentage - Third Class', ['skyblue', 'lightpink'])
plt.legend(labels, title="Gender")

plt.tight_layout()
plt.show()