# Importing the libraries and packages
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
from dash import html
import numpy as np

def createApp(store_ids, calendar_df, predictions):
    predictions = np.array(predictions)
    # creating the dash application
    app = dash.Dash('Sales Forecast')

    # creating options to list in the dropdown with the list of stores to choose from
    options = []
    for idx in range(len(store_ids)):
        options.append({'label': store_ids[idx], 'value': idx})

    # including the components in the app layout - title, dropdown label, dropdown and graph
    app.layout = html.Div([
        html.H1('Sales Forecast Application', style = {'text-align':'center'}),
        html.Label('Choose a store to predict their sales forecast'),
        dcc.Dropdown(
            id='store',
            options=options,
            value=0
        ),
        dcc.Graph(id='sale-prediction')
    ], style={'width': '500', 'font-family': 'sans-serif'})

    # callback method which is invoked when the value of the dropdown changes
    @app.callback(Output('sale-prediction', 'figure'), [Input('store', 'value')])
    def update_graph(selected_dropdown_value):
        return {
            'data': [{
                'x': calendar_df[1880:1941]['date'],
                'y': predictions[:, selected_dropdown_value]
            }],
            'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
        }

    return app