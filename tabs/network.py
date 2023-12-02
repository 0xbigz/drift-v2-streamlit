from datetime import datetime as dt
import sys
from tokenize import tabsize
import driftpy
import pandas as pd
import numpy as np
from driftpy.accounts.oracle import *
from constants import ALL_MARKET_NAMES
import plotly.express as px

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_user_account,
    get_state_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.drift_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.addresses import *
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime
import networkx as nx
import plotly.graph_objs as go
from  PIL import Image

async def get_user_stats(clearing_house: DriftClient):
    ch = clearing_house
    all_user_stats = await ch.program.account["UserStats"].all()
    user_stats_df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    return user_stats_df


async def show_network(clearing_house: DriftClient):
    url = "https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/"
    url += "market/%s/trades/%s/%s/%s"
    mol1, molselect, mol0, mol2, _ = st.columns([3, 3, 3, 3, 10])
    market_name = mol1.selectbox(
        "market",
        ALL_MARKET_NAMES,
    )
    range_selected = molselect.selectbox("range select:", ["daily", "weekly"], 0)

    dates = []
    lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)
    if range_selected == "daily":
        date = mol0.date_input(
            "select approx. date:",
            lastest_date,
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
        dates = [date]
    elif range_selected == "weekly":
        start_date = mol0.date_input(
            "start date:",
            lastest_date - datetime.timedelta(days=7),
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
        end_date = mol2.date_input(
            "end date:",
            lastest_date,
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
        dates = pd.date_range(start_date, end_date)
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, day) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (market_name, year, month, day)
        data_urls.append(data_url)
        dfs.append(pd.read_csv(data_url))
    dfs = pd.concat(dfs)
    dd, _ = st.columns([6, 5])
    with dd.expander("data sources"):
        st.write(data_urls)


    #Create the network graph using networkx

    dfs = dfs[dfs.taker.isin(dfs.groupby('taker').count().iloc[:,0].sort_values().index.tolist()[-10:])]
    if len(dfs):     
        col1, col2, col3, col4, col5, col6 = st.columns( [1, 1, 1, 1, 1, 1])
        with col1:
            layout= st.selectbox('Choose a network layout',('Random Layout','Spring Layout','Shell Layout','Kamada Kawai Layout','Spectral Layout'))
        with col2:
            color=st.selectbox('Choose color of the nodes', ('Blue','Red','Green','Orange','Red-Blue','Yellow-Green-Blue'))      
        with col3:
            title=st.text_input('Add a chart title')
        with col4:
            source=st.selectbox('Choose "source" column', ['taker', 'maker', 'filler', 'actionExplanation'])      
        with col5:
            target=st.selectbox('Choose "target" column', [ 'maker', 'taker', 'filler', 'actionExplanation'])   
        with col6:
            weight=st.selectbox('Choose "weight" column', [ 'quoteAssetAmountFilled', 'baseAssetAmountFilled', 'fillerReward', 'quoteAssetAmountSurplus', 'takerFee'])         
        # source = 'maker'
        # target = 'taker'
        df=dfs
        A = list(df[source].unique())
        B = list(df[target].unique())
        node_list = set(A+B)
        G = nx.Graph() #Use the Graph API to create an empty network graph object
        
        #Add nodes and edges to the graph object
        for i in node_list:
            G.add_node(i)
        for i,j in df.iterrows():
            G.add_weighted_edges_from([(j[source],j[target], j[weight])])
    
        #Create three input widgets that allow users to specify their preferred layout and color schemes


        #Get the position of each node depending on the user' choice of layout
        if layout=='Random Layout':
            pos = nx.random_layout(G) 
        elif layout=='Spring Layout':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif  layout=='Shell Layout':
            pos = nx.shell_layout(G)            
        elif  layout=='Kamada Kawai Layout':
            pos = nx.kamada_kawai_layout(G) 
        elif  layout=='Spectral Layout':
            pos = nx.spectral_layout(G) 

        #Use different color schemes for the node colors depending on he user input
        if color=='Blue':
            colorscale='blues'    
        elif color=='Red':
            colorscale='reds'
        elif color=='Green':
            colorscale='greens'
        elif color=='Orange':
            colorscale='orange'
        elif color=='Red-Blue':
            colorscale='rdbu'
        elif color=='Yellow-Green-Blue':
            colorscale='YlGnBu'

        #Add positions of nodes to the graph
        for n, p in pos.items():
            G.nodes[n]['pos'] = p


        #Use plotly to visualize the network graph created using NetworkX
        #Adding edges to plotly scatter plot and specify mode='lines'
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1,color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        #Adding nodes to plotly scatter plot
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale=colorscale, #The color scheme of the nodes will be dependent on the user's input
                color=[],
                size=20,
                colorbar=dict(
                    thickness=10,
                    title='# Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=0)))

        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        for node, adjacencies in enumerate(G.adjacency()):
            node_trace['marker']['color']+=tuple([len(adjacencies[1])]) #Coloring each node based on the number of connections 
            node_info = str(adjacencies[0]) +' # of connections: '+str(len(adjacencies[1]))
            node_trace['text']+=tuple([node_info])
        
        #Plot the final figure
        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title, #title takes input from the user
                        title_x=0.45,
                        titlefont=dict(size=25),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        st.plotly_chart(fig, use_container_width=True) #Show the graph in streamlit