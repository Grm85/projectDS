import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import os

# Load the data
path = r"C:\Users\LENOVO\OneDrive\Desktop\ChurnAnalysis\Prediction_Data.xlsx"
assert os.path.isfile(path), "The specified file does not exist."
df = pd.read_excel(path)

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the Dash application
app.layout = html.Div([
    html.H1("Customer Churn Analysis Dashboard"),
    
    dcc.Dropdown(
        id='x-axis-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='Tenure',  # Default value for x-axis
        clearable=False
    ),
    
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='Monthly_Charge',  # Default value for y-axis
        clearable=False
    ),
    
    dcc.Graph(id='churn-scatter-plot'),

    html.Div(id='data-summary')
])

# Callback to update the scatter plot based on selected axes
@app.callback(
    Output('churn-scatter-plot', 'figure'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value')
)
def update_scatter_plot(x_axis, y_axis):
    if x_axis in df.columns and y_axis in df.columns:
        figure = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f'Scatter Plot of {y_axis} vs {x_axis}',
            hover_data=['Customer_ID', 'Churn_Category']  # You can include other relevant columns here
        )
        return figure
    else:
        return px.scatter()  # Return an empty figure if columns are not found

# Optional: Callback to summarize data when the application starts
@app.callback(
    Output('data-summary', 'children'),
    Input('x-axis-dropdown', 'value')
)
def display_data_summary(x_axis):
    # Display a summary of the DataFrame (you can customize this further)
    return f"Displaying data for {x_axis}."

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)














