import numpy as np
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import math
from scipy.fft import fft

pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url)
df = df.dropna()
df['Date'] = pd.to_datetime(df['Country/Region'], format='%m/%d/%y')

#%%
my_app = dash.Dash('My app', external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = my_app.server
my_app.layout = html.Div([
    html.H1('DashBoard', style={'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id='questions', children=[
        dcc.Tab(label='COVID-19 Cases', value='Q1'),
        dcc.Tab(label='Quadratic Function Plotter', value='Q2'),
        dcc.Tab(label='Calculator', value='Q3'),
        dcc.Tab(label='Polynomial', value='Q4'),
        dcc.Tab(label='FFT', value='Q5'),
        dcc.Tab(label='Two-Layer Neural Network', value='Q6'),
    ]),
    html.Div(id='layout')
])

Q1_layout = html.Div([
    html.H1("COVID-19 Cases"),
    html.H5('Pick the country Name'),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in ['US', 'Brazil', 'United Kingdom', 'China', 'India', 'Italy', 'Germany']],
        multi=True,
        placeholder="Pick the country Name"
    ),
    dcc.Graph(id='covid-graph')
])

@my_app.callback(
    Output('covid-graph', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_graph(selected_countries):
    if not selected_countries:
        return go.Figure()

    filtered_df = df[['Date'] + selected_countries].copy()
    melted_df = filtered_df.melt(id_vars='Date', value_vars=selected_countries, var_name='Country', value_name='Cases')

    fig = px.line(melted_df, x='Date', y='Cases', color='Country', title='COVID-19 Global Confirmed Cases')
    return fig

Q2_layout = html.Div([
    html.H1("Quadratic Function Plotter"),
    html.Label("Select coefficients for the quadratic function:"),
    html.Div([
        html.Label("a:"),
        dcc.Slider(id='a-slider', min=-10, max=10, step=0.5, value=1, marks={i: str(i) for i in range(-10, 11)}),
    ]),
    html.Div([
        html.Label("b:"),
        dcc.Slider(id='b-slider', min=-10, max=10, step=0.5, value=0, marks={i: str(i) for i in range(-10, 11)}),
    ]),
    html.Div([
        html.Label("c:"),
        dcc.Slider(id='c-slider', min=-10, max=10, step=0.5, value=0, marks={i: str(i) for i in range(-10, 11)}),
    ]),
    dcc.Graph(id='quadratic-plot')
])

@my_app.callback(
    Output('quadratic-plot', 'figure'),
    [Input('a-slider', 'value'), Input('b-slider', 'value'), Input('c-slider', 'value')]
)
def update_plot(a, b, c):
    x = np.linspace(-10, 10, 1000)
    y = a * x**2 + b * x + c
    figure = {
        'data': [go.Scatter(x=x, y=y, mode='lines', name='Quadratic Function')],
        'layout': go.Layout(title='Plot of the Quadratic Function', xaxis={'title': 'x'}, yaxis={'title': f'f(x) = {a}x² + {b}x + {c}'})
    }
    return figure

Q3_layout = html.Div([
    html.H1("Calculator", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Please enter the first number"),
        dcc.Input(id='input-a', type='number', placeholder='Input1', style={'marginBottom': '10px', 'width': '100%'}),
        html.Label("Please select an operation"),
        dcc.Dropdown(
            id='operation-dropdown',
            options=[
                {'label': '+', 'value': 'add'},
                {'label': '-', 'value': 'subtract'},
                {'label': '*', 'value': 'multiply'},
                {'label': '/', 'value': 'divide'},
                {'label': 'log', 'value': 'log'},
                {'label': 'Root', 'value': 'root'}
            ],
            placeholder='Select an operation',
            style={'marginBottom': '10px', 'width': '100%'}
        ),
        html.Label("Please enter the second number"),
        dcc.Input(id='input-b', type='number', placeholder='Input2', style={'marginBottom': '10px', 'width': '100%'}),
    ], style={'maxWidth': '500px', 'margin': '0 auto'}),
    html.Button('Calculate', id='calculate-button', n_clicks=0, style={'marginTop': '10px', 'marginBottom': '10px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto'}),
    html.Div(id='result', style={'marginTop': '20px', 'textAlign': 'center'})
])

@my_app.callback(
    Output('result', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [State('input-a', 'value'), State('input-b', 'value'), State('operation-dropdown', 'value')]
)
def update_result(n_clicks, a, b, operation):
    if n_clicks > 0:
        if a is None or (b is None and operation not in ['log', 'root']):
            return "Please enter both input values."
        if operation == 'add':
            return f"The output value is {a + b}"
        elif operation == 'subtract':
            return f"The output value is {a - b}"
        elif operation == 'multiply':
            return f"The output value is {a * b}"
        elif operation == 'divide':
            if b == 0:
                return "Error: Division by zero is not allowed."
            return f"The output value is {a / b}"
        elif operation == 'log':
            if a <= 0 or b <= 0 or b == 1:
                return "Error: 'a' must be positive, 'b' must be positive and not equal to 1 for logarithm."
            return f"The output value is {math.log(a, b)}"
        elif operation == 'root':
            if b == 0:
                return "Error: Cannot calculate zeroth root."
            if a < 0 and b % 2 == 0:
                return "Error: Cannot take even root of a negative number."
            return f"The output value is {a ** (1 / b)}"
        else:
            return "Please select a valid operation."
    return ""

Q4_layout = html.Div([
    html.H1("Polynomial"),
    html.H5('Please enter the polynomial order', style={'font-weight': 'bold'}),
    dcc.Input(id='polynomial-order', type='number', value=1, min=1, max=10, step=1),
    dcc.Graph(id='polynomial-plot')
])

@my_app.callback(
    Output('polynomial-plot', 'figure'),
    [Input('polynomial-order', 'value')]
)
def update_polynomial_plot(order):
    if order is None or order < 1:
        order = 1
    x = np.linspace(-2, 2, 1000)
    y = x ** int(order)
    trace = go.Scatter(x=x, y=y, mode='lines', line=dict(color='blue'))
    layout = go.Layout(
        xaxis={'title': 'x'},
        yaxis={'title': f'x^{int(order)}'},
        margin={'l': 40, 'b': 40, 't': 20, 'r': 20},
        plot_bgcolor='rgba(240,240,240,0.9)'
    )
    figure = go.Figure(data=[trace], layout=layout)
    return figure

Q5_layout = html.Div([
    html.H1("FFT"),
    html.Label('Please enter the number of sinusoidal cycles'),
    dcc.Input(id='num_cycles', type='number', value=4, min=1),
    html.Label('Please enter the mean of the white noise'),
    dcc.Input(id='mean_noise', type='number', value=0),
    html.Label('Please enter the standard deviation of the white noise'),
    dcc.Input(id='std_noise', type='number', value=1, min=0),
    html.Label('Please enter the number of samples'),
    dcc.Input(id='num_samples', type='number', value=1000, min=1),
    dcc.Graph(id='sin_noise_graph'),
    html.H4('The fast Fourier transform of above generated data'),
    dcc.Graph(id='fft_graph')
])

@my_app.callback(
    [Output('sin_noise_graph', 'figure'), Output('fft_graph', 'figure')],
    [Input('num_cycles', 'value'), Input('mean_noise', 'value'), Input('std_noise', 'value'), Input('num_samples', 'value')]
)
def update_graphs(num_cycles, mean_noise, std_noise, num_samples):
    if num_cycles is None or num_cycles <= 0:
        num_cycles = 1
    if num_samples is None or num_samples <= 0:
        num_samples = 1000
    if std_noise is None or std_noise < 0:
        std_noise = 1
    x = np.linspace(0, 2 * np.pi, int(num_samples))
    noise = np.random.normal(mean_noise, std_noise, int(num_samples))
    y = np.sin(num_cycles * x) + noise

    trace1 = go.Scatter(x=x, y=y, mode='lines', name='f(x) = sin(x) + noise')
    layout1 = go.Layout(xaxis={'title': 'x'}, yaxis={'title': 'y'})
    fig1 = go.Figure(data=[trace1], layout=layout1)

    y_fft = fft(y)
    freq = np.fft.fftfreq(len(y), d=(x[1] - x[0]))
    idx = np.argsort(freq)
    trace2 = go.Scatter(x=freq[idx], y=np.abs(y_fft[idx]), mode='lines', name='FFT')
    layout2 = go.Layout(xaxis={'title': 'Frequency'}, yaxis={'title': 'Amplitude'})
    fig2 = go.Figure(data=[trace2], layout=layout2)

    return fig1, fig2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def two_layer_nn(p, W_1_1_1, W_1_1_2, b_1_1, W_2_1_1, b_2_1, W_1_2_2, b_1_2):
    a1_1 = sigmoid(p * W_1_1_1 + b_1_1)
    a1_2 = sigmoid(p * W_1_1_2 + b_2_1)
    a2 = W_2_1_1 * a1_1 + W_1_2_2 * a1_2 + b_1_2
    return a2

Q6_layout = html.Div([
    html.H1("Two-Layer Neural Network"),
    html.Label("Adjust the weights and biases for the neural network:"),
    html.Div([
        html.Img(id='neural_network_image', src='assets/neural_network_diagram.png',
                 style={'width': '900px', 'height': '500px'})
    ]),
    dcc.Graph(id='nn-plot'),

    html.Div([
        html.Label("b₁¹:"),
        dcc.Slider(id='b_1_1-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        html.Label("b₂¹:"),
        dcc.Slider(id='b_2_1-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        html.Label("W₁₁¹:"),
        dcc.Slider(id='W_1_1_1-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        html.Label("W₁₁²:"),
        dcc.Slider(id='W_1_1_2-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        html.Label("b₁²:"),
        dcc.Slider(id='b_1_2-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        html.Label("W₂₁¹:"),
        dcc.Slider(id='W_2_1_1-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div([
        html.Label("W₁₂²:"),
        dcc.Slider(id='W_1_2_2-slider', min=-10, max=10, step=0.1, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),
], style={'display': 'flex', 'flex-direction': 'column'})

@my_app.callback(
    Output('nn-plot', 'figure'),
    [Input('b_1_1-slider', 'value'),
     Input('b_2_1-slider', 'value'),
     Input('W_1_1_1-slider', 'value'),
     Input('W_1_1_2-slider', 'value'),
     Input('b_1_2-slider', 'value'),
     Input('W_2_1_1-slider', 'value'),
     Input('W_1_2_2-slider', 'value')]
)
def update_graph(b_1_1, b_1_2, W_1_1_1, W_1_1_2, b_2_1, W_2_1_1, W_1_2_2):
    p = np.linspace(-5, 5, 1000)
    a2 = two_layer_nn(p, W_1_1_1, W_1_1_2, b_1_1, W_2_1_1, b_2_1, W_1_2_2, b_1_2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=a2, mode='lines', name='a2'))
    fig.update_layout(title='Output a2 vs Input p',
                      xaxis_title='Input p',
                      yaxis_title='Output a2')
    return fig



@my_app.callback(
    Output('layout', 'children'),
    [Input('questions', 'value')]
)
def update_layout(question):
    if question == 'Q1':
        return Q1_layout
    elif question == 'Q2':
        return Q2_layout
    elif question == 'Q3':
        return Q3_layout
    elif question == 'Q4':
        return Q4_layout
    elif question == 'Q5':
        return Q5_layout
    else:
        return Q6_layout

if __name__ == '__main__':
    my_app.run_server(debug=True, host='0.0.0.0', port=8080)