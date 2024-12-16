import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.linalg import svd, cond
pd.set_option('display.float_format', '{:.2f}'.format)
#%%
print('1')
stocks = px.data.stocks()

print("list of Features:", stocks.columns.tolist())
print("\n Last 5 Observations:")
print(stocks.tail())


#%%
print('2')
fig = go.Figure()
for company in stocks.columns[1:]:
    fig.add_trace(go.Scatter(
        x=stocks['date'],
        y=stocks[company],
        mode='lines',
        name=company,
        line=dict(width=4)
    ))

fig.update_layout(
    title=dict(text='Stock Prices of Companies',
               font=dict(color='red', family='Times New Roman', size=30),
               x=0.5, xanchor='center'),
    xaxis=dict(title=dict(text='Time', font=dict(color='yellow', family='Courier New', size=30)),
               tickfont=dict(family='Courier New', color='yellow', size=30)),
    yaxis=dict(title=dict(text='Normalized ($)', font=dict(color='yellow', family='Courier New', size=30)),
               tickfont=dict(family='Courier New', color='yellow', size=30)),
    legend=dict(title=dict(text='Variable', font=dict(color='green', size=30)),
                font=dict(family='Courier New', color='yellow', size=30)),
    width=2000,
    height=800,
    template='plotly_dark'
)
fig.show(renderer='browser')

#%%
print('3')
stocks = px.data.stocks()
fig = make_subplots(rows = 3, cols = 2)
companies = stocks.columns[1:]
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
for i, company in enumerate(companies):
    row = (i // 2) + 1
    col = (i % 2) + 1
    fig.add_trace(
        go.Histogram(x = stocks[company], nbinsx=50, name = company, marker_color=colors[i]),
        row = row,
        col = col
    )
fig.update_layout(
    title=dict(text='Histogram Plot',
               font=dict(color='red', family='Times New Roman', size=30),
               x=0.5, xanchor='center'),
    legend=dict(title=dict(text='Variable', font=dict(color='green', size=30)),
                font=dict(family='Courier New', color='black', size=30)),
    template='seaborn'
)
for i in range(1, 4):
    for j in range(1, 3):
        fig.update_xaxes(title_text='Normalized Price ($)', row=i, col=j, title_font=dict(size=15, color='black'), tickfont=dict(size=15, color='black'))
        fig.update_yaxes(title_text='Frequency', row=i, col=j, title_font=dict(size=15, color='black'), tickfont=dict(size=15, color='black'))
fig.show(renderer='browser')

#%% PCA Analysis
print('4')
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd, cond
scaler = StandardScaler()
features = stocks.iloc[:,1:]
print(features.columns)
scaled_features = scaler.fit_transform(features)

#%%
u, s, vh = svd(scaled_features, full_matrices=False)
condition_number = cond(scaled_features)
print("Singular Values:", s.round(2))
print("Condition Number:", condition_number.round(2))

#%%
correlation_matrix = features.corr()
plt.figure(figsize = (10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, center=0, cbar_kws={'shrink': 0.75})
plt.title('Correlation Coefficient between features - Original feature space')
plt.show()

#%%
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(scaled_features)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(explained_variance_ratio >= 0.95) + 1

print("Number of components needed to reach 95% explained variance:", n_components_95)
print("Explained Variance Ratio (Original Feature Space):", pca.explained_variance_ratio_[:n_components_95].round(2))
print("Explained Variance Ratio (Reduced Feature Space):", explained_variance_ratio[:n_components_95].round(2))

#%%
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, marker='o', linestyle='--', color='b')
plt.axhline(y=95, color='r', linestyle='--', label='95% Explained Variance')
plt.axvline(x=n_components_95, color='k', linestyle='--', label=f'Optimal Number of Components: {n_components_95}')
plt.xlabel('Number of Components', fontsize=15)
plt.ylabel('Cumulative Explained Variance (%)', fontsize=15)
plt.title('Cumulative Explained Variance vs Number of Components', fontsize=18)
plt.legend()
plt.grid()
plt.show()


#%%
reduced_features = pca.transform(scaled_features)[:, :n_components_95]
u_reduced, s_reduced, vh_reduced = svd(reduced_features, full_matrices=False)
condition_number_reduced = cond(reduced_features)

print("Singular Values (Reduced Feature Space):", s_reduced.round(2))
print("Condition Number (Reduced Feature Space):", condition_number_reduced.round(2))

#%%
reduced_features_df = pd.DataFrame(reduced_features, columns=[f'PC{i+1}' for i in range(n_components_95)])
reduced_correlation_matrix = reduced_features_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(reduced_correlation_matrix, annot=True, cmap='RdPu', linewidths=0.5, center=0, cbar_kws={'shrink': 0.75})
plt.title('Correlation Coefficient between features - Reduced feature space')
plt.show()

#%%
new_columns = [f'Principal col {i + 1}' for i in range(n_components_95)]
reduced_features_df = pd.DataFrame(reduced_features, columns=new_columns)
print("\nFirst 5 rows of the newly created DataFrame")
print(reduced_features_df.head())

#%%
fig = go.Figure()
for col in reduced_features_df.columns:
    fig.add_trace(go.Scatter(
        x=stocks['date'],
        y=reduced_features_df[col],
        mode='lines',
        name=col,
        line=dict(width=4)
    ))

fig.update_layout(
    title=dict(text='PCA Transformed Features vs Time',
               font=dict(color='red', family='Times New Roman', size=30),
               x=0.5, xanchor='center'),
    xaxis=dict(title=dict(text='Time', font=dict(color='yellow', family='Courier New', size=30)),
               tickfont=dict(family='Courier New', color='yellow', size=30)),
    yaxis=dict(title=dict(text='PCA Transformed Value', font=dict(color='yellow', family='Courier New', size=30)),
               tickfont=dict(family='Courier New', color='yellow', size=30)),
    legend=dict(title=dict(text='Variable', font=dict(color='green', size=30)),
                font=dict(family='Courier New', color='yellow', size=30)),
    width=2000,
    height=800,
    template='plotly_dark'
)
fig.show(renderer='browser')

#%%
fig = make_subplots(rows=int(n_components_95), cols=1)
for i, col in enumerate(reduced_features_df.columns):
    fig.add_trace(
        go.Histogram(x=reduced_features_df[col], nbinsx=50, name=col, marker_color=colors[i % len(colors)]),
        row=i+1, col=1
    )

fig.update_layout(
    title=dict(text='Histogram Plot of PCA Transformed Features',
               font=dict(color='red', family='Times New Roman', size=30),
               x=0.5, xanchor='center'),
    legend=dict(title=dict(text='Variable', font=dict(color='green', size=30)),
                font=dict(family='Courier New', color='black', size=30)),
    template='seaborn'
)
for i in range(1, n_components_95 + 1):
    fig.update_xaxes(title_text='Price($)', row=i, col=1, title_font=dict(size=15, color='black'), tickfont=dict(size=15, color='black'))
    fig.update_yaxes(title_text='Frequency', row=i, col=1, title_font=dict(size=15, color='black'), tickfont=dict(size=15, color='black'))
fig.show(renderer='browser')

#%%
print('k')
fig_original = px.scatter_matrix(
    stocks,
    dimensions=stocks.columns[1:],
    title='Original Feature Space',
    template='seaborn',
    width=1000,
    height=1000
)
fig_original.update_traces(diagonal_visible=False)
fig_original.update_layout(
    title=dict(font=dict(color='black', family='Times New Roman', size=20))
)
fig_original.show(renderer='browser')

fig_reduced = px.scatter_matrix(
    reduced_features_df,
    dimensions=reduced_features_df.columns,
    title='Reduced Feature Space',
    template='seaborn',
    width=1000,
    height=1000
)
fig_reduced.update_traces(diagonal_visible=False)
fig_reduced.update_layout(
    title=dict(font=dict(color='black', family='Times New Roman', size=20))
)
fig_reduced.show(renderer='browser')

