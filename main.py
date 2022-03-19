# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16bs_AL9meSP25LUOvTDTTwKEreRSWmBH
"""

# pip install dash==0.31.1  # The core dash backend
# pip install dash-html-components==0.13.2  # HTML components
# pip install dash-core-components==0.39.0  # Supercharged components
# pip install dash-table==3.1.7  # Interactive DataTable component (new!)
#
# pip install plotly==5.2.2

import dash
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

# Data Exploration with Pandas (python)
# -----------------------------------------------------------------

df = pd.read_csv("Totaldatasheet9.csv")  # data by GregorySmith from kaggle
# -----------------------------------------------------------------


app = dash.Dash(__name__)

in_click = None

app.layout = html.Div([
    html.H1('Interactive dashboard for visualisation and assessment of hydrological parameters'),
    dcc.Dropdown(id='parameters-choice', options=[{'label': x, 'value': x} for x in df.columns],value='Rainfall_mm'),
    dcc.Graph(id='my-graph', figure={}),
    html.Button("Download CSV", id="btn_csv"),
    dcc.Download(id="download-dataframe-csv"),
])

@app.callback(
    [Output(component_id='my-graph', component_property='figure'),Output("download-dataframe-csv", "data")],
    [Input(component_id='parameters-choice', component_property='value'),
     Input(component_id="btn_csv", component_property="n_clicks")],
)


def interactive_graphing(value, n_clicks=0):
    global in_click
    if value == 'Rainfall_mm':
        fig = px.bar(data_frame=df, x='Dates', y='Rainfall_mm')
        print(n_clicks)
        if n_clicks != in_click:
            in_click = n_clicks
            dff = df[['Dates', value]]
            return fig, dcc.send_data_frame(dff.to_csv, str(value) + '.csv')
        return fig, None
    elif value == 'Evapotranspiration_mm':
        fig = px.bar(df, x='Date_8', y='Evapotranspiration_ Eta_mm_MODIS_MOD16A2')
        if n_clicks != in_click:
            in_click = n_clicks
            dff = df[['Date_8', value]]
            return fig, dcc.send_data_frame(dff.to_csv, str(value) + '.csv')
        return fig, None
    elif value == 'Root_zone_soil_moisture_mm_FLDAS':
        fig = px.bar(data_frame=df, x='Date_monthly', y='Root_zone_soil_moisture_mm_FLDAS')
        if n_clicks != in_click:
            in_click = n_clicks
            dff = df[['Date_monthly', value]]
            return fig, dcc.send_data_frame(dff.to_csv, str(value) + '.csv')
        return fig, None
    elif value == 'Current_year_storage_BCM_Total':
        fig = px.bar(df, x='Date_weekly', y='Current_year_storage_BCM_Total')
        if n_clicks != in_click:
            in_click = n_clicks
            dff = df[['Date_weekly', value]]
            return fig, dcc.send_data_frame(dff.to_csv, str(value) + '.csv')
        return fig, None
    elif value == 'Total_Runoff_mm_FLDAS':
        fig = px.bar(df, x='Date_monthly', y='Total_Runoff_mm_FLDAS')
        if n_clicks != in_click:
            in_click = n_clicks
            dff = df[['Date_monthly', value]]
            return fig, dcc.send_data_frame(dff.to_csv, str(value) + '.csv')
        return fig, None


if __name__ == '__main__':
    app.run_server(debug=False)
