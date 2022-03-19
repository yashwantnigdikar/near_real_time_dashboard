import dash
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

# Data Exploration with Pandas (python)
# -----------------------------------------------------------------

df = pd.read_csv("file.csv")  # data by GregorySmith from kaggle
# -----------------------------------------------------------------

sheet_id = '1NqL2E1ufxrP7TbgrnPkIEMK5z1ZPTVjVsRfZbXnCRsM'

df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
print(df.columns)

app = dash.Dash(__name__)

in_click = None

app.layout = html.Div([html.Div([
            html.H1('Dashboard for visualization of parameters of water budgeting'),
            html.H2('parameter-one'),
            dcc.Dropdown(id='parameters-one', options=[{'label': x, 'value': x} for x in df.columns],value='Dates')]),
        html.Div([html.H2('parameter-two'),
            dcc.Dropdown(id='parameters-two', options=[{'label': x, 'value': x} for x in df.columns],value='Rainfall_mm'),
            dcc.Graph(id='my-graph', figure={}),
            html.Button("Download CSV", id="btn_csv"),
            dcc.Download(id="download-dataframe-csv")]),
])

@app.callback(
    [Output(component_id='my-graph', component_property='figure'),Output("download-dataframe-csv", "data")],
    [Input(component_id='parameters-one', component_property='value'),
     Input(component_id='parameters-two', component_property='value'),
     Input(component_id="btn_csv", component_property="n_clicks")],
)


def interactive_graphing(value1, value2, n_clicks=0):
    global in_click
    fig = px.bar(data_frame=df, x=value1, y=value2)
    print(n_clicks)
    if n_clicks != in_click:
        in_click = n_clicks
        dff = df[['Dates', value2]]
        return fig, dcc.send_data_frame(dff.to_csv, str(value2) + '.csv')
    return fig, None



if __name__ == '__main__':
    app.run_server(debug=False)
