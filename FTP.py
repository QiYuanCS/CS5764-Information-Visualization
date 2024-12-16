import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from numpy.linalg import svd, cond
from scipy import stats
from scipy.stats import probplot
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.float_format', '{:.2f}'.format)


# X, Y label settings
title_font = {'family': 'serif', 'color': 'blue', 'size': 20}
label_font = {'family': 'serif', 'color': 'darkred', 'size': 16}

pd.set_option('display.max_columns', None)
#%%
import pandas as pd
from prettytable import PrettyTable

df = pd.read_csv('Invistico_Airline.csv')

table = PrettyTable()
table.field_names = df.columns.tolist()

for _, row in df.head(25).iterrows():
    table.add_row(row.tolist())

print(table)
#%%
df = pd.read_csv('Invistico_Airline.csv')

categorical_features = ['satisfaction', 'Gender', 'Customer Type', 'Type of Travel', 'Class']
numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Service-related features
service_features = ['Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
                    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                    'Inflight entertainment', 'On-board service', 'Leg room service',
                    'Baggage handling', 'Checkin service', 'Cleanliness']

numerical_features = numerical_features + service_features



df_cleaned = df.dropna(inplace=True)
all_numerical_features = numerical_features
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values by dropping rows with any missing values
df_cleaned = df.dropna()

# Convert categorical variables to appropriate data types
for feature in categorical_features:
    df_cleaned[feature] = df_cleaned[feature].astype('category')

# Check for outliers in numerical features using IQR
Q1 = df_cleaned[numerical_features].quantile(0.25)
Q3 = df_cleaned[numerical_features].quantile(0.75)
IQR = Q3 - Q1

# Define outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
for feature in numerical_features:
    df_cleaned = df_cleaned[(df_cleaned[feature] >= lower_bound[feature]) & (df_cleaned[feature] <= upper_bound[feature])]

print("Data cleaning completed.")

print(df_cleaned.head())

#%%


def create_numerical_subplots(df, features, plot_type, palette='Set2', y=None):
    # Handle specific plot types that require multiple features
    if plot_type in ['cluster_map', '3d_plot', 'contour_plot', 'heatmap','jointplot']:
        if plot_type == 'cluster_map':
            # Generate a cluster map of the correlation matrix of numerical features
            cluster_data = df[features].dropna()
            corr_matrix = cluster_data.corr()
            cg = sns.clustermap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
            cg.fig.suptitle('Cluster Map for Numerical Features', y=1.05)
            plt.show()
        elif plot_type == 'heatmap':
            heatmap_data = df[features].dropna().corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title('Heatmap of Numerical Features Correlation', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        elif plot_type == '3d_plot':
            # Ensure there are at least 3 numerical features
            if len(features) >= 3:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    df[features[0]],
                    df[features[1]],
                    df[features[2]],
                    c='blue',
                    alpha=0.6,
                    edgecolor='k'
                )
                ax.set_title(f'3D Scatter Plot for {features[0]}, {features[1]}, {features[2]}')
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                plt.show()
            else:
                print("3D plot requires at least 3 numerical features. Skipping.")

        elif plot_type == 'contour_plot':
            # Ensure there are at least 3 numerical features
            if len(features) >= 3:
                x = df[features[0]]
                y_data = df[features[1]]
                z = df[features[2]]
                # Remove NaN values
                mask = (~x.isna()) & (~y_data.isna()) & (~z.isna())
                x, y_data, z = x[mask], y_data[mask], z[mask]
                # Create grid values
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y_data.min(), y_data.max(), 100)
                xi, yi = np.meshgrid(xi, yi)
                # Interpolate to get zi values
                zi = griddata((x, y_data), z, (xi, yi), method='linear')
                plt.figure(figsize=(10, 6))
                contour = plt.contourf(xi, yi, zi, cmap="viridis", alpha=0.8)
                plt.colorbar(contour, label=features[2])
                plt.title(f'Contour Plot for {features[0]}, {features[1]}, and {features[2]}')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.show()
            else:
                print("Contour plot requires at least 3 numerical features. Skipping.")
        elif plot_type == 'jointplot':
            # Joint plot with KDE and scatter representation
            if len(features) >= 2:
                x = features[0]
                y_feature = features[1]
                if y is not None and y in df.columns and df[y].dtype in ['object', 'category']:
                    sns.jointplot(data=df, x=x, y=y_feature, kind='scatter', hue=y, palette=palette, height=8, alpha=0.6)
                else:
                    sns.jointplot(data=df, x=x, y=y_feature, kind='scatter', height=8, alpha=0.6)
                plt.suptitle(f'Joint Plot between {x} and {y_feature}', fontsize=16)
                plt.subplots_adjust(top=0.95)  # 调整以为 suptitle 留出空间
                plt.show()
            else:
                print("Joint plot requires at least 2 numerical features. Skipping.")

    else:
        # For other plot types, create subplots
        # Determine the number of rows based on the number of features
        num_features = len(features)
        num_cols = 2
        num_rows = (num_features + 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 6 * num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            if i >= len(axes):
                break  # Prevent indexing errors

            ax = axes[i]
            if plot_type == 'line':
                sns.lineplot(data=df, x=df.index, y=feature, ax=ax, marker='o', color='blue')
                ax.set_title(f'Line Plot for {feature}')
                ax.set_xlabel('Index')
                ax.set_ylabel(feature)

            elif plot_type == 'dist':
                sns.histplot(df[feature].dropna(), kde=False, bins=30, color='blue', ax=ax)
                ax.set_title(f'Distribution of {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')

            elif plot_type == 'hist_kde':
                sns.histplot(df[feature].dropna(), kde=True, stat="density", linewidth=0, ax=ax, color='green')
                ax.set_title(f'Histogram with KDE for {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')

            elif plot_type == 'qq':
                probplot(df[feature].dropna(), dist="norm", plot=ax)
                ax.set_title(f'QQ Plot for {feature}')
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')

            elif plot_type == 'kde':
                sns.kdeplot(data=df[feature].dropna(), fill=True, alpha=0.6, linewidth=2, ax=ax, color='green')
                ax.set_title(f'KDE Plot for {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')

            elif plot_type in ['box', 'violin']:
                if y is not None and (df[y].dtype == 'object' or df[y].dtype.name == 'category'):
                    if plot_type == 'box':
                        sns.boxplot(data=df, x=y, y=feature, ax=ax, palette=palette)
                        ax.set_title(f'Box Plot of {feature} by {y}')
                    elif plot_type == 'violin':
                        sns.violinplot(data=df, x=y, y=feature, ax=ax, palette=palette)
                        ax.set_title(f'Violin Plot of {feature} by {y}')
                    ax.set_xlabel(y)
                    ax.set_ylabel(feature)
                else:
                    ax.set_visible(False)
                    continue

            elif plot_type == 'reg':
                if y is not None and pd.api.types.is_numeric_dtype(df[y]):
                    sns.regplot(data=df, x=feature, y=y, ax=ax, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
                    ax.set_title(f'Regression Plot for {feature} vs {y}')
                    ax.set_xlabel(feature)
                    ax.set_ylabel(y)
                else:
                    ax.set_visible(False)
                    continue

            elif plot_type == 'rug':
                sns.rugplot(data=df, x=feature, ax=ax, height=0.1, color='red')
                ax.set_title(f'Rug Plot for {feature}')
                ax.set_xlabel(feature)
            else:
                ax.set_visible(False)

            ax.grid(True)

        # Remove any unused subplots
        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

# 1. Line Plots
create_numerical_subplots(df_cleaned, all_numerical_features, 'line', palette='Set2')

# 2. Distribution Plots
create_numerical_subplots(df_cleaned, all_numerical_features, 'dist', palette='Set3')

# 3. Histogram with KDE
create_numerical_subplots(df_cleaned, all_numerical_features, 'hist_kde', palette='Set1')

# 4. QQ Plots
create_numerical_subplots(df_cleaned, all_numerical_features, 'qq')

# 5. KDE Plots
create_numerical_subplots(df_cleaned, all_numerical_features, 'kde', palette='Dark2')

# 6. Regression Plots
# Choose a numerical target variable for regression, e.g., 'Age'
create_numerical_subplots(df_cleaned, all_numerical_features, 'reg', palette='Accent', y='Age')

# 7. Box Plots
# Ensure 'satisfaction' is categorical
create_numerical_subplots(df_cleaned, all_numerical_features, 'box', y='satisfaction')


# 9. Violin Plots
create_numerical_subplots(df_cleaned, all_numerical_features, 'violin', y='satisfaction', palette='Pastel1')

# 10. Rug Plots
create_numerical_subplots(df_cleaned, all_numerical_features, 'rug')

# 11. Cluster Map
create_numerical_subplots(df_cleaned, all_numerical_features, 'cluster_map')

three_features = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']  # Adjust as needed
create_numerical_subplots(df_cleaned, three_features, '3d_plot')

three_features_contour = three_features  # Adjust as needed
create_numerical_subplots(df_cleaned, three_features_contour, 'contour_plot')


#%%
# Define sample size for sampling plots
DEFAULT_SAMPLE_SIZE = 500
def create_subplots(df, features, plot_type, y=None, palette='Set2', bar_type='grouped', sample_size=DEFAULT_SAMPLE_SIZE):

    num_features = len(features)
    cols = 2
    rows = (num_features + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        # Convert palette name to color array if provided
        if palette is not None:
            palette_colors = sns.color_palette(palette)
        else:
            palette_colors = None

        # Sampling for swarm and strip plots to improve performance
        if plot_type in ['swarm', 'strip'] and len(df) > sample_size:
            df_sampled = df.sample(sample_size, random_state=5764)
        else:
            df_sampled = df

        # Plot based on the specified plot type
        if plot_type == 'bar':
            if bar_type == 'grouped':
                sns.countplot(data=df, x=feature, ax=ax, palette=palette_colors)
            elif bar_type == 'stacked':
                if y is None:
                    raise ValueError("y parameter must be provided for stacked bar plots.")
                crosstab = pd.crosstab(df[feature], df[y])
                crosstab.plot(kind='bar', stacked=True, ax=ax, color=palette_colors)
                ax.legend(title=y)
            else:
                raise ValueError("bar_type must be either 'grouped' or 'stacked'.")
            ax.set_title(f'Distribution of {feature}', fontsize=16)
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel('Count', fontsize=14)

        elif plot_type == 'count':
            sns.countplot(data=df, x=feature, ax=ax, palette=palette_colors)
            ax.set_title(f'Count Plot of {feature}', fontsize=16)
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel('Count', fontsize=14)

        elif plot_type == 'pie':
            feature_counts = df[feature].value_counts()
            if palette is not None:
                palette_colors_pie = sns.color_palette(palette, len(feature_counts))
            else:
                palette_colors_pie = None
            ax.pie(
                feature_counts,
                labels=feature_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=palette_colors_pie
            )
            ax.set_title(f'Pie Chart of {feature}', fontsize=16)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        elif plot_type == 'strip':
            if y is None:
                raise ValueError("y parameter must be provided for strip plots.")
            sns.stripplot(data=df_sampled, x=feature, y=y, ax=ax, palette=palette_colors, jitter=True)
            ax.set_title(f'Strip Plot of {feature} vs {y}', fontsize=16)
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel(y, fontsize=14)

        elif plot_type == 'swarm':
            if y is None:
                raise ValueError("y parameter must be provided for swarm plots.")
            sns.swarmplot(data=df_sampled, x=feature, y=y, ax=ax, palette=palette_colors)
            ax.set_title(f'Swarm Plot of {feature} vs {y}', fontsize=16)
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel(y, fontsize=14)

        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Creating grouped and stacked bar subplots
print("\nCreating grouped bar subplots:")
create_subplots(
    df=df_cleaned,
    features=categorical_features,
    plot_type='bar',
    y='satisfaction',
    palette='Set3',
    bar_type='grouped',
    sample_size=DEFAULT_SAMPLE_SIZE
)

print("\nCreating stacked bar subplots:")
create_subplots(
    df=df_cleaned,
    features=categorical_features,
    plot_type='bar',
    y='satisfaction',
    palette='Set3',
    bar_type='stacked',
    sample_size=DEFAULT_SAMPLE_SIZE
)

# Creating count subplots
print("\nCreating count subplots:")
create_subplots(
    df=df_cleaned,
    features=categorical_features,
    plot_type='count',
    palette='Paired',
    sample_size=DEFAULT_SAMPLE_SIZE
)

# Creating pie subplots
print("\nCreating pie subplots:")
create_subplots(
    df=df_cleaned,
    features=categorical_features,
    plot_type='pie',
    palette='Pastel1',
    sample_size=DEFAULT_SAMPLE_SIZE
)

# Creating strip subplots
print("\nCreating strip subplots:")
create_subplots(
    df=df_cleaned,
    features=categorical_features,
    plot_type='strip',
    y='satisfaction',
    palette='Dark2',
    sample_size=DEFAULT_SAMPLE_SIZE
)

# Creating swarm subplots
print("\nCreating swarm subplots:")
create_subplots(
    df=df_cleaned,
    features=categorical_features,
    plot_type='swarm',
    y='satisfaction',
    palette='Accent',
    sample_size=DEFAULT_SAMPLE_SIZE
)

#%%GCP DashBoard
#%%
import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from flask_caching import Cache
from scipy.stats import gaussian_kde
from scipy.stats import shapiro, kstest, anderson
from scipy import stats
from sklearn.decomposition import PCA
import plotly.colors as pc
import matplotlib as plt
import hashlib

df = pd.read_csv('Invistico_Airline.csv')
categorical_features = ['satisfaction', 'Gender', 'Customer Type', 'Type of Travel', 'Class']
numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

service_features = ['Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
                    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                    'Inflight entertainment', 'On-board service', 'Leg room service',
                    'Baggage handling', 'Checkin service', 'Cleanliness']

numerical_features = numerical_features + service_features
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.cssapp']


app = dash.Dash('My app', external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.title = 'Interactive Data Visualization Dashboard'
server = app.server

cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

common_style = {'marginBottom': '20px'}
div_style = {'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderTop': '1px solid #ddd'}
dropdown_style = {'width': '50%', 'marginBottom': '20px'}

app.layout = html.Div([
    dcc.Store(id='original-dataset', data=df.to_json(date_format='iso', orient='split')),
    dcc.Store(id='cleaned-dataset'),
    dcc.Store(id='outlier-processed-data', storage_type='memory'),
    dcc.Store(id='processed-data'),

    html.H1('Interactive Data Visualization Dashboard', style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='tab1', children=[
            dcc.Tab(label='Numerical Features', value='tab2'),
            dcc.Tab(label='Categorical Features', value='tab3'),
        ], style={'fontSize': '16px'}),
        html.Div(id='tabs-content', style={'padding': '20px'})
    ], style={'width': '100%', 'display': 'block', 'padding': '20px', 'marginBottom': '30px'}),

    html.Div([
        html.H2('Data Processing Options', style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            html.H3('Missing Value Processing Options'),
            html.Label('Missing Value Method for Numerical Features:'),
            dcc.Dropdown(
                id='missing-value-method-num',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': "Mode", 'value': 'mode'},
                ],
                value='mean',
                style={'marginBottom': '20px'}
            ),

            html.Label('Missing Value Method for Categorical Features:'),
            dcc.Dropdown(
                id='missing-value-method-cat',
                options=[
                    {'label': 'Mode', 'value': 'mode'},
                ],
                value='mode',
                style={'marginBottom': '20px'}
            ),

            html.Button('Pre-process', id='preprocess-button', n_clicks=0, style={'marginTop': '10px'}),
            html.Div(id='preprocess-message', style={'marginTop': '10px', 'color': 'green'})
        ], style={'marginBottom': '30px'}),

        html.Div([
            html.H3('Cleaned Dataset (The First Few Observations)'),
            dash_table.DataTable(
                id='cleaned-data-table',
                style_table={'overflowX': 'auto'},
                page_size=10
            )
        ], style={'width': '100%', 'padding': '20px', 'marginBottom': '30px'}),

        html.Div([
            html.H3('Statistics of Cleaned Dataset'),
            dash_table.DataTable(
                id='statistics-data-table',
                style_table={'overflowX': 'auto'},
                page_size=10
            )
        ], style={'width': '100%', 'padding': '20px', 'marginBottom': '30px'}),
        html.H3('Data Process', style=common_style),

        html.Div([
            html.H3('Outlier Detection Section', style=common_style),
            html.Div([
                html.Label('Select Outlier Detection Method:'),
                dcc.Dropdown(
                    id='outlier-detection-method',
                    options=[
                        {'label': 'Z-Score', 'value': 'zscore'},
                        {'label': 'IQR', 'value': 'iqr'}
                    ],
                    value='zscore',
                    style=dropdown_style
                ),
                html.Label('Set Threshold:'),
                dcc.Input(
                    id='outlier-threshold',
                    type='number',
                    value=3,
                    min=0,
                    step=0.1,
                    style=common_style
                ),
                html.Button('Detect Outliers', id='detect-outliers-button', n_clicks=0, style=common_style),
                html.Div(id='outlier-detection-message', style={'color': 'red'}),
            ]),
            dash_table.DataTable(
                id='outlier-detection-table',
                columns=[],
                data=[],
                page_size=10
            )
        ], style=div_style),

        html.Div([
            html.H3('Normality Test', style=common_style),
            html.Div([
                html.Label('Select Normality Test Method:'),
                dcc.Dropdown(
                    id='normality-test-method',
                    options=[
                        {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
                        {'label': 'Kolmogorov-Smirnov Test', 'value': 'kstest'},
                        {'label': 'Anderson-Darling Test', 'value': 'anderson'}
                    ],
                    value='shapiro',
                    style=dropdown_style
                ),
                html.Button(
                    'Run Normality Test',
                    id='run-normality-test-button',
                    n_clicks=0,
                    style=common_style
                ),
            ]),
            dash_table.DataTable(
                id='normality-test-table',
                columns=[{'name': 'Test Statistic', 'id': 'Test Statistic'},
                         {'name': 'P-value', 'id': 'P-value'}],
                data=[],
                page_size=10
            )
        ], style=div_style),

        html.Div([
            html.Label('Data Transformation:', style={'marginBottom': '10px'}),
            dcc.Dropdown(
                id='scaling-method',
                options=[
                    {'label': 'Standardization', 'value': 'standardization'},
                    {'label': 'Normalization', 'value': 'normalization'},
                    {'label': 'None', 'value': 'none'}
                ],
                value='standardization',
                style=dropdown_style
            ),
            html.Div([
                html.Button('Confirm', id='confirm-button', n_clicks=0, style={'marginRight': '10px'}),
                html.Button('Reset', id='reset-button', n_clicks=0)
            ], style={'marginTop': '20px'}),
            html.Div(id='processing-message', style={'marginTop': '20px', 'color': 'green'})
        ], style=div_style),
        dash_table.DataTable(
            id='processed-data-table',
            columns=[],
            data=[],
            page_size=10
        ),
    ], style={'width': '100%', 'padding': '20px'}),
    html.H3('Feature Selection using Random Forest', style={'marginBottom': '20px'}),
    html.Div([
        html.Button('Run Random Forest Feature Selection', id='run-rf-button', n_clicks=0,
                    style={'marginBottom': '20px'}),
        html.Div(id='rf-message', style={'color': 'green'}),

        dcc.Graph(id='feature-importance-bar-chart'),

        dcc.Graph(id='cumulative-importance-line-chart'),

        html.H4('Selected Features'),
        dash_table.DataTable(
            id='selected-features-table',
            columns=[{'name': 'Feature', 'id': 'Feature'}, {'name': 'Importance', 'id': 'Importance'}],
            data=[],
            page_size=10
        ),

        html.H4('Reduced Dataset Preview'),
        dash_table.DataTable(
            id='reduced-dataset-table',
            columns=[],
            data=[],
            page_size=10
        )
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderTop': '1px solid #ddd'}),
    html.H3('Heatmap & Pearson Correlation Coefficient Matrix', style={'marginBottom': '20px'}),
    html.Div([
        html.Button('Generate Correlation Analysis', id='generate-correlation-button', n_clicks=0,
                    style={'marginBottom': '20px'}),
        html.Div(id='correlation-message', style={'color': 'green'}),

        dcc.Graph(id='correlation-heatmap'),

        dcc.Graph(id='scatter-plot-matrix')
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderTop': '1px solid #ddd'}),
    html.H3('Statistical Analysis', style={'marginBottom': '20px'}),
    html.Div([
        html.Button('Perform Statistical Analysis', id='perform-stat-analysis-button', n_clicks=0,
                    style={'marginBottom': '20px'}),
        html.Div(id='stat-analysis-message', style={'color': 'green'}),

        html.H4('Statistical Measures'),
        dash_table.DataTable(
            id='statistical-measures-table',
            columns=[],
            data=[],
            page_size=10
        ),

        dcc.Dropdown(
            id='kde-feature-x',
            options=[],
            placeholder='Select X-axis Feature',
            style={'width': '45%', 'display': 'inline-block', 'marginRight': '10px'}
        ),
        dcc.Dropdown(
            id='kde-feature-y',
            options=[],
            placeholder='Select Y-axis Feature',
            style={'width': '45%', 'display': 'inline-block'}
        ),
        html.Button('Generate KDE Plot', id='generate-kde-button', n_clicks=0, style={'marginTop': '20px'}),
        dcc.Graph(id='multivariate-kde-plot')
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderTop': '1px solid #ddd'})
])


def handle_missing_values(df, numerical_features, categorical_features,
                          missing_value_method_num='mean', missing_value_method_cat='mode'):

    df_filled = df.copy()

    if missing_value_method_num == 'mean':
        df_filled[numerical_features] = df_filled[numerical_features].fillna(df_filled[numerical_features].mean())
    elif missing_value_method_num == 'median':
        df_filled[numerical_features] = df_filled[numerical_features].fillna(df_filled[numerical_features].median())

    if missing_value_method_cat == 'mode':
        df_filled[categorical_features] = df_filled[categorical_features].fillna(df_filled[categorical_features].mode().iloc[0])

    return df_filled

@app.callback(
    [Output('cleaned-dataset', 'data'),
     Output('preprocess-message', 'children')],
    [Input('preprocess-button', 'n_clicks')],
    [State('original-dataset', 'data'),
     State('missing-value-method-num', 'value'),
     State('missing-value-method-cat', 'value')]
)
def execute_preprocessing(n_clicks, original_data, missing_value_method_num, missing_value_method_cat):
    if n_clicks and original_data:
        df = pd.read_json(original_data, orient='split')
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

        df_cleaned = handle_missing_values(df, numerical_features, categorical_features,
                                           missing_value_method_num=missing_value_method_num,
                                           missing_value_method_cat=missing_value_method_cat)
        cleaned_data = df_cleaned.to_json(date_format='iso', orient='split')
        message = 'Pre-process Finished'
        return cleaned_data, message
    return None, ''

@app.callback(
    [Output('cleaned-data-table', 'columns'),
     Output('cleaned-data-table', 'data')],
    [Input('cleaned-dataset', 'data')]
)
def update_cleaned_data_table(cleaned_data):
    if cleaned_data:
        df_cleaned = pd.read_json(cleaned_data, orient='split')
        print("Clean Data Table:", df_cleaned.head(5))
        columns = [{"name": i, "id": i} for i in df_cleaned.columns]
        data = df_cleaned.head(30).to_dict('records')
        return columns, data
    else:
        print("None")
        return [], []


@app.callback(
    [Output('statistics-data-table', 'columns'),
     Output('statistics-data-table', 'data')],
    [Input('cleaned-dataset', 'data')]
)
def update_statistics_data_table(cleaned_data):
    if cleaned_data:
        df_cleaned = pd.read_json(cleaned_data, orient='split')
        stats = df_cleaned.describe(include='all').reset_index()
        print("Statistics Data Table:", stats.head())
        columns = [{"name": i, "id": i} for i in stats.columns]
        data = stats.head(30).to_dict('records')
        return columns, data
    else:
        print("None")
        return [], []

def detect_outliers(df, numerical_cols, method='zscore', threshold=3):

    df_outlier_removed = df.copy()

    if method == 'zscore':
        z_scores = np.abs((df_outlier_removed[numerical_cols] - df_outlier_removed[numerical_cols].mean()) / df_outlier_removed[numerical_cols].std())
        df_outlier_removed = df_outlier_removed[(z_scores <= threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = df_outlier_removed[numerical_cols].quantile(0.25)
        Q3 = df_outlier_removed[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        df_outlier_removed = df_outlier_removed[~((df_outlier_removed[numerical_cols] < (Q1 - 1.5 * IQR)) | (df_outlier_removed[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_outlier_removed

@app.callback(
    [Output('outlier-processed-data', 'data'),
     Output('outlier-detection-message', 'children'),
     Output('outlier-detection-table', 'columns'),
     Output('outlier-detection-table', 'data')],
    [Input('detect-outliers-button', 'n_clicks')],
    [State('processed-data', 'data'),
     State('cleaned-dataset', 'data'),
     State('outlier-detection-method', 'value'),
     State('outlier-threshold', 'value')]
)
def detect_outliers_callback(n_clicks, processed_data, cleaned_data, method, threshold):
    if n_clicks and (processed_data or cleaned_data):
        data_json = processed_data if processed_data else cleaned_data
        df = pd.read_json(data_json, orient='split')
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        df_outlier_removed = detect_outliers(df, numerical_cols, method=method, threshold=threshold)

        data_json = df_outlier_removed.to_json(date_format='iso', orient='split')
        message = f' {method.upper()} Method and Threshold {threshold} Removed Outliers'

        columns = [{"name": i, "id": i} for i in df_outlier_removed.columns]
        data = df_outlier_removed.head(30).to_dict('records')

        return data_json, message, columns, data
    else:
        return None, 'None', [], []

def run_normality_test(df, numerical_features, test='shapiro'):

    results = []
    for feature in numerical_features:
        x = df[feature].dropna()
        if test == 'shapiro':
            stat, p = stats.shapiro(x)
            result_text = f'Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p:.4f}'
        elif test == 'kstest':
            stat, p = stats.kstest(x, 'norm', args=(np.mean(x), np.std(x)))
            result_text = f'Kolmogorov-Smirnov Test: Statistic={stat:.4f}, p-value={p:.4f}'
        elif test == 'anderson':
            result = stats.anderson(x)
            result_text = f'Anderson-Darling Test: Statistic={result.statistic:.4f}'
        else:
            result_text = 'Invalid Test'
        results.append({
            'Feature': feature,
            'Test': test.capitalize(),
            'Result': result_text
        })
    return results

@app.callback(
    [Output('normality-test-table', 'columns'),
     Output('normality-test-table', 'data')],
    [Input('run-normality-test-button', 'n_clicks')],
    [State('processed-data', 'data'),
     State('cleaned-dataset', 'data'),
     State('normality-test-method', 'value')]
)
def run_normality_test_callback(n_clicks, processed_data, cleaned_data, selected_test):
    if n_clicks and (processed_data or cleaned_data):
        data_json = processed_data if processed_data else cleaned_data
        df = pd.read_json(data_json, orient='split')

        numerical_features = df.select_dtypes(include=np.number).columns.tolist()

        test_results = run_normality_test(df, numerical_features, test=selected_test)

        columns = [{"name": col, "id": col} for col in ['Feature', 'Test', 'Result']]
        data = test_results

        return columns, data
    else:
        return [], []


def process_data(df,
                 missing_value_method_num='mean',
                 missing_value_method_cat='mode',
                 scaling_method='standardization'):
    """
    通过处理缺失值和特征缩放来处理数据集。
    """
    categorical_features = ['satisfaction', 'Gender', 'Customer Type', 'Type of Travel', 'Class']
    numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

    service_features = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                        'Inflight entertainment', 'On-board service', 'Leg room service',
                        'Baggage handling', 'Checkin service', 'Cleanliness']

    numerical_features = numerical_features + service_features

    df_processed = encode_features(df, categorical_features)
    if scaling_method == 'standardization':
        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    elif scaling_method == 'normalization':
        scaler = MinMaxScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    return df_processed

@app.callback(
    [Output('processed-data', 'data'),
     Output('processing-message', 'children'),
     Output('processed-data-table', 'columns'),
     Output('processed-data-table', 'data')],
    [Input('confirm-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('outlier-processed-data', 'data'),
     State('missing-value-method-num', 'value'),
     State('missing-value-method-cat', 'value'),
     State('scaling-method', 'value')]
)
def process_and_store_data(confirm_n_clicks, reset_n_clicks, outlier_processed_data,
                           missing_value_method_num, missing_value_method_cat,
                           scaling_method):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, '', [], []
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'confirm-button' and outlier_processed_data:
        df = pd.read_json(outlier_processed_data, orient='split')
        df_processed = process_data(df,
                                    missing_value_method_num=missing_value_method_num,
                                    missing_value_method_cat=missing_value_method_cat,
                                    scaling_method=scaling_method)
        data_json = df_processed.to_json(date_format='iso', orient='split')
        message = 'Transformation Finished'

        columns = [{"name": i, "id": i} for i in df_processed.columns]
        data = df_processed.head(30).to_dict('records')

        return data_json, message, columns, data
    elif button_id == 'reset-button':
        return None, 'Data Process Reset', [], []
    else:
        return dash.no_update, '', [], []



@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('outlier-processed-data', 'data')]
)
def render_content(tab, data):
    if data is not None:
        df_processed = pd.read_json(data, orient='split')
    else:
        df_processed = df.copy()

    if tab == 'tab2':
        numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            return html.Div([
                html.H3('Numerical Feature Visualization'),
                html.Label('Select Feature:'),
                dcc.Dropdown(
                    id='numerical-feature-dropdown',
                    options=[{'label': col, 'value': col} for col in numerical_cols],
                    value=numerical_cols[0],
                    multi=False
                ),
                html.Label('Select Plot Type:'),
                dcc.RadioItems(
                    id='numerical-plot-type',
                    options=[
                        {'label': 'Line Plot', 'value': 'line'},
                        {'label': 'Distribution Plot', 'value': 'dist'},
                        {'label': 'Histogram with KDE', 'value': 'hist_kde'},
                        {'label': 'QQ Plot', 'value': 'qq'},
                        {'label': 'Regression Plot', 'value': 'reg'},
                        {'label': 'Box Plot', 'value': 'box'},
                        {'label': 'Area Plot', 'value': 'area'},
                        {'label': 'Violin Plot', 'value': 'violin'},
                        {'label': 'Joint Plot', 'value': 'joint'},
                        {'label': 'Rug Plot', 'value': 'rug'},
                        {'label': '3D Plot', 'value': '3d'},

                    ],
                    value='line',
                    labelStyle={'display': 'inline-block'}
                ),
                html.Div([
                    html.Label('Select Y Feature:', id='y-feature-label', style={'display': 'none'}),
                    dcc.Dropdown(
                        id='y-feature-dropdown',
                        options=[{'label': col, 'value': col} for col in numerical_cols],
                        value=numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0],
                        multi=False,
                        style={'display': 'none'}
                    ),
                    html.Label('Select Z Feature:', id='z-feature-label', style={'display': 'none'}),
                    dcc.Dropdown(
                        id='z-feature-dropdown',
                        options=[{'label': col, 'value': col} for col in numerical_cols],
                        value=numerical_cols[2] if len(numerical_cols) > 2 else numerical_cols[0],
                        multi=False,
                        style={'display': 'none'}
                    )
                ], id='additional-inputs'),
                dcc.Graph(id='numerical-feature-graph')
            ])
        else:
            return html.Div('No numerical features available.')
    elif tab == 'tab3':
        categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            return html.Div([
                html.H3('Categorical Feature Visualization'),
                dcc.Dropdown(
                    id='categorical-feature-dropdown',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    value=categorical_cols[0],
                    multi=False
                ),
                dcc.RadioItems(
                    id='categorical-plot-type',
                    options=[
                        {'label': 'Grouped Bar Plot', 'value': 'bar_grouped'},
                        {'label': 'Stacked Bar Plot', 'value': 'bar_stacked'},
                        {'label': 'Count Plot', 'value': 'count'},
                        {'label': 'Pie Chart', 'value': 'pie'},
                        {'label': 'Strip Plot', 'value': 'strip'},
                        {'label': 'Swarm Plot', 'value': 'swarm'},
                    ],
                    value='bar_grouped',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Graph(id='categorical-feature-graph')
            ])
        else:
            return html.Div('No categorical features available.')

@app.callback(
    [Output('y-feature-dropdown', 'style'),
     Output('y-feature-label', 'style'),
     Output('z-feature-dropdown', 'style'),
     Output('z-feature-label', 'style')],
    [Input('numerical-plot-type', 'value')]
)

def show_hide_additional_inputs(plot_type):
    if plot_type in ['reg', 'joint']:
        style_visible = {'display': 'block'}
        style_hidden = {'display': 'none'}
        return style_visible, style_visible, style_hidden, style_hidden
    elif plot_type in ['3d']:
        style_visible = {'display': 'block'}
        return style_visible, style_visible, style_visible, style_visible
    else:
        style_hidden = {'display': 'none'}
        return style_hidden, style_hidden, style_hidden, style_hidden

def encode_features(df, categorical_features):
    df_encoded = df
    le = LabelEncoder()
    for col in categorical_features:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

def train_random_forest(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=5764)
    rf.fit(X, y)

    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    feature_importances['Cumulative Importance'] = feature_importances['Importance'].cumsum()

    return feature_importances

def select_important_features(feature_importances, threshold=0.95):
    selected_features = feature_importances[feature_importances['Cumulative Importance'] <= threshold]
    return selected_features

def create_feature_importance_bar_chart(feature_importances):
    fig = px.bar(
        feature_importances,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        labels={'Importance': 'Importance', 'Feature': 'Feature'}
    )
    fig.update_layout(
        yaxis={
            'categoryorder': 'total ascending',
            'automargin': True
        },
        margin=dict(l=300),
        height=800
    )
    return fig

def create_cumulative_importance_line_chart(feature_importances):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(feature_importances) + 1)),
        y=feature_importances['Cumulative Importance'],
        mode='lines+markers',
        name='Cumulative Importance'
    ))
    fig.add_hline(y=0.95, line_dash='dash', line_color='red', annotation_text='95% Threshold')
    fig.update_layout(
        title='Cumulative Feature Importance',
        xaxis_title='Number of Features',
        yaxis_title='Cumulative Importance'
    )
    return fig

results_cache = {}

@app.callback(
    [
        Output('rf-message', 'children'),
        Output('feature-importance-bar-chart', 'figure'),
        Output('cumulative-importance-line-chart', 'figure'),
        Output('selected-features-table', 'data'),
        Output('reduced-dataset-table', 'columns'),
        Output('reduced-dataset-table', 'data')
    ],
    Input('run-rf-button', 'n_clicks'),
    State('processed-data', 'data')
)
def run_random_forest_feature_selection(n_clicks, processed_data):
    if n_clicks > 0 and processed_data:
        data_hash = hashlib.md5(processed_data.encode('utf-8')).hexdigest()
        if data_hash in results_cache:
            return results_cache[data_hash]

        df = pd.read_json(processed_data, orient='split')

        categorical_features = ['satisfaction','Gender', 'Customer Type', 'Type of Travel', 'Class']
        numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

        service_features = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                            'Inflight entertainment', 'On-board service', 'Leg room service',
                            'Baggage handling', 'Checkin service', 'Cleanliness']

        numerical_features = numerical_features + service_features

        feature_importances = train_random_forest(df, 'satisfaction')
        selected_features = select_important_features(feature_importances)

        bar_chart = create_feature_importance_bar_chart(feature_importances)
        line_chart = create_cumulative_importance_line_chart(feature_importances)

        reduced_df = df[selected_features['Feature'].tolist()]
        reduced_df['satisfaction'] = df['satisfaction']

        selected_features_data = selected_features[['Feature', 'Importance']].to_dict('records')
        reduced_columns = [{'name': col, 'id': col} for col in reduced_df.columns]
        reduced_data = reduced_df.head(30).to_dict('records')

        message = 'Random Forest feature selection completed.'

        results = (message, bar_chart, line_chart, selected_features_data, reduced_columns, reduced_data)
        results_cache[data_hash] = results
        return results
    else:
        return '', {}, {}, [], [], []

def compute_correlation_matrix(df):
    correlation_matrix = df.corr()
    return correlation_matrix

def create_correlation_heatmap(correlation_matrix):
    """
    绘制相关矩阵的热力图，包含目标变量 satisfaction。
    """
    fig = px.imshow(
        correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlation Heatmap'
    )
    fig.update_layout(
        width=1200, height=1200,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    return fig


def create_scatter_plot_matrix(df):
    if 'satisfaction' in df.columns:
        df['satisfaction'] = df['satisfaction'].map({0: 'dissatisfied', 1: 'satisfied'})
    else:
        raise ValueError("The target variable 'satisfaction' is not present in the dataset.")

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    features = numerical_features[:6]

    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color='satisfaction',
        title='Scatter Plot Matrix with Top Features'
    )
    fig.update_layout(
        width=1200,
        height=1200,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    return fig



@app.callback(
    [
        Output('correlation-message', 'children'),
        Output('correlation-heatmap', 'figure'),
        Output('scatter-plot-matrix', 'figure')
    ],
    Input('generate-correlation-button', 'n_clicks'),
    State('reduced-dataset-table', 'data'),
    State('reduced-dataset-table', 'columns')
)
def generate_correlation_analysis(n_clicks, reduced_data, reduced_columns):
    if n_clicks > 0 and reduced_data:
        df = pd.DataFrame(reduced_data, columns=[col['name'] for col in reduced_columns])

        if 'satisfaction' not in df.columns:
            raise ValueError("The target variable 'satisfaction' is not present in the reduced data.")

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        if df.empty:
            raise ValueError("No data available after processing. Please check your data and preprocessing steps.")

        correlation_matrix = compute_correlation_matrix(df)

        heatmap_fig = create_correlation_heatmap(correlation_matrix)

        scatter_matrix_fig = create_scatter_plot_matrix(df)

        message = 'Correlation analysis completed.'

        return message, heatmap_fig, scatter_matrix_fig
    else:
        return '', {}, {}

def compute_statistical_measures(df):
    stats = df.describe().T
    stats['Skewness'] = df.skew()
    stats['Kurtosis'] = df.kurtosis()
    return stats.reset_index().rename(columns={'index': 'Feature'})


@app.callback(
    [
        Output('kde-feature-x', 'options'),
        Output('kde-feature-y', 'options')
    ],
    Input('perform-stat-analysis-button', 'n_clicks'),
    State('reduced-dataset-table', 'columns')
)
def update_kde_feature_options(n_clicks, reduced_columns):
    if n_clicks > 0 and reduced_columns:
        options = [{'label': col['name'], 'value': col['name']} for col in reduced_columns]
        return options, options
    else:
        return [], []


@app.callback(
    [
        Output('stat-analysis-message', 'children'),
        Output('statistical-measures-table', 'columns'),
        Output('statistical-measures-table', 'data')
    ],
    Input('perform-stat-analysis-button', 'n_clicks'),
    State('reduced-dataset-table', 'data'),
    State('reduced-dataset-table', 'columns')
)
def perform_statistical_analysis(n_clicks, reduced_data, reduced_columns):
    if n_clicks > 0 and reduced_data:
        df = pd.DataFrame(reduced_data, columns=[col['name'] for col in reduced_columns])

        stats = compute_statistical_measures(df)

        columns = [{'name': i, 'id': i} for i in stats.columns]
        data = stats.head(30).to_dict('records')

        message = 'Statistical analysis completed.'

        return message, columns, data
    else:
        return '', [], []


@app.callback(
    Output('multivariate-kde-plot', 'figure'),
    Input('generate-kde-button', 'n_clicks'),
    State('kde-feature-x', 'value'),
    State('kde-feature-y', 'value'),
    State('reduced-dataset-table', 'data'),
    State('reduced-dataset-table', 'columns')
)
def generate_multivariate_kde(n_clicks, feature_x, feature_y, reduced_data, reduced_columns):
    if n_clicks > 0 and feature_x and feature_y and reduced_data:
        df = pd.DataFrame(reduced_data, columns=[col['name'] for col in reduced_columns])

        kde_fig = create_multivariate_kde(df, feature_x, feature_y)

        return kde_fig
    else:
        return {}


def create_multivariate_kde(df, feature_x, feature_y):
    fig = px.density_contour(
        df,
        x=feature_x,
        y=feature_y,
        color='satisfaction',
        title=f'Multivariate KDE of {feature_x} vs {feature_y}'
    )
    fig.update_layout(width=1200, height=1200)
    return fig

def plot_line(df, feature):
    fig = px.line(df, x=df.index, y=feature, title=f'Line Plot of {feature}')
    return fig

def plot_dist(df, feature):
    fig = ff.create_distplot([df[feature].dropna()], [feature], show_hist=False)
    fig.update_layout(title=f'Distribution Plot of {feature}')
    return fig

def plot_pair(df, features):
    fig = px.scatter_matrix(df[features])
    fig.update_layout(title='Pair Plot')
    return fig

def plot_heatmap(df, features):
    corr = df[features].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis'
    ))
    fig.update_layout(title='Heatmap with Color Bar')
    return fig

def plot_hist_kde(df, feature):
    x = df[feature].dropna()

    kde = gaussian_kde(x)
    x_grid = np.linspace(x.min(), x.max(), 1000)
    y_kde = kde.evaluate(x_grid)

    hist = go.Histogram(
        x=x,
        histnorm='probability density',
        nbinsx=30,
        opacity=0.7,
        name='Histogram'
    )

    kde_line = go.Scatter(
        x=x_grid,
        y=y_kde,
        mode='lines',
        line=dict(color='red', width=2),
        name='KDE'
    )

    fig = go.Figure(data=[hist, kde_line])
    fig.update_layout(
        title=f'Histogram with KDE of {feature}',
        xaxis_title=feature,
        yaxis_title='Density',
        bargap=0.2
    )
    return fig



def plot_qq(df, feature):
    import scipy.stats as stats
    import numpy as np
    sample_data = df[feature].dropna()
    (osm, osr), (slope, intercept, r) = stats.probplot(sample_data, dist="norm", plot=None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'))
    # Add a reference line
    line = slope * osm + intercept
    fig.add_trace(go.Scatter(x=osm, y=line, mode='lines', name='Fit'))
    fig.update_layout(
        title=f'QQ Plot of {feature}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Ordered Values',
        showlegend=False
    )
    return fig


def plot_reg(df, x_feature, y_feature):
    fig = px.scatter(df, x=x_feature, y=y_feature, trendline='ols')
    fig.update_layout(title=f'Regression Plot of {x_feature} vs {y_feature}')
    return fig

def plot_box(df, feature, by_feature=None):
    if by_feature:
        fig = px.box(df, x=by_feature, y=feature)
        fig.update_layout(title=f'Box Plot of {feature} by {by_feature}')
    else:
        fig = px.box(df, y=feature)
        fig.update_layout(title=f'Box Plot of {feature}')
    return fig

def plot_area(df, feature):
    fig = px.area(df, x=df.index, y=feature, title=f'Area Plot of {feature}')
    return fig

def plot_violin(df, feature, by_feature=None):
    if by_feature:
        fig = px.violin(df, x=by_feature, y=feature, box=True, points='all')
        fig.update_layout(title=f'Violin Plot of {feature} by {by_feature}')
    else:
        fig = px.violin(df, y=feature, box=True, points='all')
        fig.update_layout(title=f'Violin Plot of {feature}')
    return fig

def plot_joint(df, x_feature, y_feature):
    fig = px.scatter(
        df, x=x_feature, y=y_feature, opacity=0.6,
        marginal_x='violin', marginal_y='violin',
        trendline='ols',
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(title=f'Joint Plot of {x_feature} and {y_feature}')
    return fig


def plot_rug(df, feature):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[feature], opacity=0.6, name='Histogram'))
    fig.add_trace(go.Scatter(
        x=df[feature],
        y=np.zeros_like(df[feature]),
        mode='markers',
        marker=dict(color='red', symbol='line-ns-open'),
        name='Rug Plot'
    ))
    fig.update_layout(title=f'Rug Plot of {feature}')
    return fig

def plot_3d(df, x_feature, y_feature, z_feature):
    fig = px.scatter_3d(df, x=x_feature, y=y_feature, z=z_feature, color=z_feature)
    fig.update_layout(title=f'3D Scatter Plot of {x_feature}, {y_feature}, {z_feature}')
    return fig


def plot_cluster_map(df, features):
    corr = df[features].corr()
    sns.clustermap(corr, cmap='coolwarm')
    plt.title('Cluster Map')
    plt.show()



@app.callback(
    Output('numerical-feature-graph', 'figure'),
    [Input('numerical-feature-dropdown', 'value'),
     Input('numerical-plot-type', 'value'),
     Input('y-feature-dropdown', 'value'),
     Input('z-feature-dropdown', 'value'),
     Input('outlier-processed-data', 'data')]
)
def update_numerical_graph(selected_feature, plot_type, y_feature, z_feature, outlier_processed_data):
    if outlier_processed_data is not None:
        df_processed = pd.read_json(outlier_processed_data, orient='split')
    else:
        df_processed = df.copy()

    if selected_feature is None or selected_feature not in df_processed.columns:
        return go.Figure()

    df_outliers_removed = df_processed

    if plot_type == 'line':
        fig = plot_line(df_outliers_removed, selected_feature)
    elif plot_type == 'dist':
        fig = plot_dist(df_outliers_removed, selected_feature)
    elif plot_type == 'hist_kde':
        fig = plot_hist_kde(df_outliers_removed, selected_feature)
    elif plot_type == 'qq':
        fig = plot_qq(df_outliers_removed, selected_feature)
    elif plot_type == 'reg':
        if y_feature and y_feature in df_outliers_removed.columns:
            fig = plot_reg(df_outliers_removed, selected_feature, y_feature)
        else:
            fig = go.Figure()
    elif plot_type == 'box':
        fig = plot_box(df_outliers_removed, selected_feature)
    elif plot_type == 'area':
        fig = plot_area(df_outliers_removed, selected_feature)
    elif plot_type == 'violin':
        fig = plot_violin(df_outliers_removed, selected_feature)
    elif plot_type == 'joint':
        if y_feature and y_feature in df_outliers_removed.columns:
            fig = plot_joint(df_outliers_removed, selected_feature, y_feature)
        else:
            fig = go.Figure()
    elif plot_type == 'rug':
        fig = plot_rug(df_outliers_removed, selected_feature)
    elif plot_type == '3d':
        if (y_feature and z_feature and
                y_feature in df_outliers_removed.columns and
                z_feature in df_outliers_removed.columns):
            fig = plot_3d(df_outliers_removed, selected_feature, y_feature, z_feature)
        else:
            fig = go.Figure()
    else:
        fig = go.Figure()

    return fig



def plot_bar(df, feature, target='satisfaction', bar_type='group'):
    counts = df.groupby([feature, target]).size().reset_index(name='count')
    if pd.api.types.is_categorical_dtype(df[feature]):
        counts[feature] = counts[feature].cat.as_ordered()
    else:
        counts = counts.sort_values(by=feature)
    fig = px.bar(
        counts,
        x=feature,
        y='count',
        color=target,
        barmode=bar_type,
        title=f'Bar Chart of {feature} by {target} ({bar_type.title()}ed)',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        xaxis_title=feature,
        yaxis_title='Count',
        legend_title=target,
        title={'x': 0.5}
    )

    return fig


def plot_count(df, feature, palette='Plotly'):
    fig = px.histogram(df, x=feature, color=feature, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(title=f'Count Plot of {feature}')
    return fig

def plot_pie(df, feature, palette='Pastel1'):
    value_counts = df[feature].value_counts().reset_index()
    value_counts.columns = [feature, 'Count']
    fig = px.pie(value_counts, names=feature, values='Count', color_discrete_sequence=px.colors.qualitative.Pastel1)
    fig.update_layout(title=f'Pie Chart of {feature}')
    return fig

def plot_strip(df, feature, y, palette='Dark2'):
    fig = px.strip(df, x=feature, y=y, color=feature, color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(title=f'Strip Plot of {feature} vs {y}')
    return fig

def plot_swarm(df, feature, y, palette='Bold'):
    fig = px.scatter(df, x=feature, y=y, color=feature, color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(title=f'Swarm Plot of {feature} vs {y}')
    return fig

def plot_histogram(df, feature):
    fig = px.histogram(df, x=feature, nbins=30, title=f'Histogram of {feature}')
    return fig

def plot_kde(df, feature):
    fig = px.density_contour(df, x=feature, title=f'KDE Plot of {feature}', color_continuous_scale='Viridis')
    return fig


@app.callback(
    Output('categorical-feature-graph', 'figure'),
    [Input('categorical-feature-dropdown', 'value'),
     Input('categorical-plot-type', 'value'),
     Input('cleaned-dataset', 'data')]
)
def update_categorical_graph(selected_feature, plot_type, data):
    if data is not None:
        df_processed = pd.read_json(data, orient='split')
    else:
        df_processed = df.copy()

    if selected_feature is None or selected_feature not in df_processed.columns:
        return {}


    if plot_type in ['bar_grouped', 'bar_stacked']:
        bar_type = 'group' if plot_type == 'bar_grouped' else 'stack'

        if selected_feature == 'satisfaction':
            return {}

        fig = plot_bar(df_processed, selected_feature, target='satisfaction', bar_type=bar_type)

    elif plot_type == 'count':
        fig = plot_count(df_processed, selected_feature)
    elif plot_type == 'pie':
        fig = plot_pie(df_processed, selected_feature)
    elif plot_type == 'strip':
        numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            y_feature = numerical_cols[0]
            fig = plot_strip(df_processed, selected_feature, y=y_feature)
        else:
            return {}
    elif plot_type == 'swarm':
        numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            y_feature = numerical_cols[0]
            fig = plot_swarm(df_processed, selected_feature, y=y_feature)
        else:
            return {}
    else:
        return {}

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)