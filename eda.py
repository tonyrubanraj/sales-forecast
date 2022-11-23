# Importing the libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def calculateAverageSales(input_df, itemList, title):
    """
    helper function to calculate average monthly sales by various groups and plot the data
    """
    
    fig = go.Figure()
    for item in itemList:
        cols = [c for c in input_df.columns if item in c]
        df = input_df[cols].sum(axis=1)
        df.columns = ['date','sum']
        df = df.groupby(pd.Grouper(freq="M")).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df, name=item))

    fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title=str("Monthy Average Sales " + "(" + title + ")"), template='plotly_white')
    fig.show()


def calculateAverageSalesByCategory(input_df, itemList, category):
    """
    helper function to calculate average monthly sales by product categories across states and plot the data    
    """
    
    fig = go.Figure()
    for item in itemList:
        cols = [c for c in input_df.columns if item in c and category in c]
        df = input_df[cols].sum(axis=1)
        df.columns = ['date','sum']
        df = df.groupby(pd.Grouper(freq="M")).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df, name=item))

    fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Monthy Average Sales (state) : " + category, template='plotly_white')
    fig.show()


def eda(calendar_df, sales_df):
    """
    Method to perform exploratory data analysis

    Arguments:
    calendar_df -- dataframe containing the calendar information
    sales_df -- dataframe containing the sales of each product in every store

    Returns:
    --
    """

    # fetching the days column from the sales_df
    days_columns = sales_df.columns[6:]
    store_ids = list(set(sales_df['store_id']))

    # merging the calendar df with the sales df to analyse the sales of products on a monthly basis
    store_df = sales_df.set_index('id')[days_columns].T.merge(calendar_df.set_index('d')['date'],
           left_index=True,
           right_index=True,
           validate='1:1').set_index('date')
    store_df.index = pd.to_datetime(store_df.index)

    # calculating monthly average sales per store
    calculateAverageSales(store_df, store_ids, "store")

    # calculating monthly average sales per state
    state_ids = set(sales_df['state_id'])
    calculateAverageSales(store_df, state_ids, "state")

    # calculating monthly average sales per category
    category_ids = set(sales_df['cat_id'])
    calculateAverageSales(store_df, category_ids, "category")

    # calculating monthly average sales of hobby products in each state
    calculateAverageSalesByCategory(store_df, state_ids, list(category_ids)[0])

    # calculating monthly average sales of foods products in each state
    calculateAverageSalesByCategory(store_df, state_ids, list(category_ids)[1])

    # calculating monthly average sales of household products in each state
    calculateAverageSalesByCategory(store_df, state_ids, list(category_ids)[2])

    # merging the calendar df with the sales df to analyse the sales of products during special events and holidays
    snap_sales_df = sales_df.set_index('id')[days_columns].T.merge(calendar_df.set_index('d')[['date','snap_CA','snap_TX','snap_WI', 'event_type_1', 'event_type_2']],
            left_index=True,
            right_index=True,
            validate='1:1').set_index('date')

    snap_sales_df.index = pd.to_datetime(snap_sales_df.index)

    # calculating monthly sales during SNAP event in each state
    fig = go.Figure()
    for state in state_ids:
        state_columns = [c for c in snap_sales_df.columns if state in c and c != ("snap_"+state)]
        df = snap_sales_df.loc[snap_sales_df['snap_'+state]==1][state_columns].sum(axis=1)
        df.columns = ['date','sum']
        df = df.groupby(pd.Grouper(freq="M")).sum() / 30
        fig.add_trace(go.Scatter(x=df.index, y=df, name=state))

    fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Monthy Average Sales during SNAP event (state)", template='plotly_white')
    fig.show()

    # analysing the sales during SNAP events opposed to remaining days
    snap_df = pd.DataFrame(columns=['State', 'Avg sales on SNAP days', 'Avg sales on remaining days'])
    for state in state_ids:
        state_columns = [c for c in snap_sales_df.columns if state in c and c != ("snap_"+state)]
        on_snap_df = snap_sales_df.loc[snap_sales_df['snap_'+state]==1][state_columns].sum(axis=1)
        not_snap_df = snap_sales_df.loc[snap_sales_df['snap_'+state]==0][state_columns].sum(axis=1)
        snap_df = pd.concat([pd.DataFrame([[state, on_snap_df.mean(), not_snap_df.mean()]], columns = snap_df.columns), snap_df], ignore_index=True)
    
    snap_df.plot.bar()
    plt.xticks(range(len(state_ids)), snap_df.State)
    plt.title('Average sales during SNAP events and remaining days')
    plt.show()

    # analysing the sales during each major holiday event
    events = set(snap_sales_df['event_type_1'])
    events.remove(np.nan)
    event_df = pd.DataFrame(columns=['Event', 'Avg sales'])
    for event in events:
        columns = snap_sales_df.columns[:-5]
        event_sales_df = snap_sales_df.loc[(snap_sales_df['event_type_1']==event)][columns].sum(axis=1)
        event_df = pd.concat([pd.DataFrame([[event, event_sales_df.mean()]], columns = event_df.columns), event_df], ignore_index=True)

    event_sales_df = snap_sales_df.loc[(snap_sales_df['event_type_1'].isnull())][columns].sum(axis=1)
    event_df = pd.concat([pd.DataFrame([['Normal days', event_sales_df.mean()]], columns = event_df.columns), event_df], ignore_index=True)
    event_df.plot.bar()
    plt.xticks(range(len(event_df.Event)), event_df.Event)
    plt.title('Average sales during holiday events')
    plt.show()

    # deleting garbage variables
    del event_sales_df
    del snap_sales_df
    del event_df