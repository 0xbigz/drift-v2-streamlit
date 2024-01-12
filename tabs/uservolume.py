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
from driftpy.constants.perp_markets import mainnet_perp_market_configs, devnet_perp_market_configs, PerpMarketConfig
from driftpy.addresses import *
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime

import asyncio
import httpx

def find_value(df, search_column, search_string, target_column):
    """
    Returns the value of a target column in a Pandas DataFrame where a search string is found in a search column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to search.
    search_column (str): The name of the column to search for the search string.
    search_string (str): The string to search for in the search column.
    target_column (str): The name of the column to return the value from.

    Returns:
    The value of the target column in the row where the search string is found in the search column. If the search
    string is not found in the search column, returns None.
    """
    # Create a boolean mask for the rows where the search string is found in the search column.
    mask = df[search_column].str.contains(search_string, na=False)

    # Use the mask to filter the DataFrame and select the target column for the matching row.
    target_value = df.loc[mask, target_column].values

    # If there are no matching rows, return None. Otherwise, return the first matching value.
    return target_value[0] if len(target_value) > 0 else None


async def get_user_stats(clearing_house: DriftClient):
    ch = clearing_house
    all_user_stats = await ch.program.account["UserStats"].all()
    user_stats_df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    return user_stats_df

from io import StringIO


async def fetch_lp_data(date, market_name):
    market_index = [x.market_index for x in mainnet_perp_market_configs if market_name == x.symbol]
    if len(market_index) == 1:
        market_index = market_index[0]
    else:
        return None, ''
    url = f'https://dlob-data.s3.eu-west-1.amazonaws.com/mainnet-beta/{date.strftime("%Y-%m-%d")}/lp-shares-{market_index}.csv.gz'
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 403:
            return None, ''
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text)), url

async def load_lp_info_async(dates, market_names, with_urls=False):
    data_urls = []
    dfs = []

    tasks = [fetch_lp_data(date, market_name) for market_name in market_names for date in dates]

    # Use asyncio.gather to run the tasks concurrently
    results = await asyncio.gather(*tasks)

    for df, data_url in results:
        if df is not None:
            df['name'] = data_url.split('/')[-1].split('-')[-1].split('.')[0]
            dfs.append(df)
            data_urls.append(data_url)

    result_df = pd.concat(dfs)

    if with_urls:
        return result_df, data_urls

    return result_df

async def fetch_volume_data(date, market_name):
    if market_name == 'JitoSOL':
        market_name = 'jitoSOL'

    url = f"https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/{market_name}/trades/{date.year}/{date.month}/{date.day}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 403:
            return None, ''
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text)), url

async def load_volumes_async(dates, market_names, with_urls=False):
    data_urls = []
    dfs = []

    tasks = [fetch_volume_data(date, market_name) for market_name in market_names for date in dates]

    # Use asyncio.gather to run the tasks concurrently
    results = await asyncio.gather(*tasks)

    for df, data_url in results:
        if df is not None:
            dfs.append(df)
            data_urls.append(data_url)

    result_df = pd.concat(dfs)

    if with_urls:
        return result_df, data_urls

    return result_df

def load_volumes(dates, market_name, with_urls=False):
    url = "https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/"
    url += "market/%s/trades/%s/%s/%s"
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, day) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (market_name, year, month, day)
        data_urls.append(data_url)
        dfs.append(pd.read_csv(data_url))
    dfs = pd.concat(dfs)

    if with_urls:
        return dfs, data_urls

    return dfs

def calc_maker_exec_price(row):
    if row["makerOrderDirection"] == "long":
        # user is long, paying more is bad, add fee to quoteAmt (negative fee is rebate)
        return (row["quoteAssetAmountFilled"] + row["makerFee"]) / row[
            "baseAssetAmountFilled"
        ]
    elif row["makerOrderDirection"] == "short":
        # user is short, recv more is good, sub fee from quoteAmt (negative fee is rebate)
        return (row["quoteAssetAmountFilled"] - row["makerFee"]) / row[
            "baseAssetAmountFilled"
        ]
    else:
        return (row["quoteAssetAmountFilled"]) / row[
            "baseAssetAmountFilled"
        ]
    # return row["quoteAssetAmountFilled"] / row["baseAssetAmountFilled"]

def calc_taker_exec_price(row):
    if row["takerOrderDirection"] == "long":
        # user is long, paying more is bad, add fee to quoteAmt (negative fee is rebate)
        return (row["quoteAssetAmountFilled"] + row["takerFee"]) / row[
            "baseAssetAmountFilled"
        ]
    elif row["takerOrderDirection"] == "short":
        # user is short, recv more is good, sub fee from quoteAmt (negative fee is rebate)
        return (row["quoteAssetAmountFilled"] - row["takerFee"]) / row[
            "baseAssetAmountFilled"
        ]
    else:
        return (row["quoteAssetAmountFilled"]) / row[
            "baseAssetAmountFilled"
        ]

    # return row["quoteAssetAmountFilled"] / row["baseAssetAmountFilled"]

def calc_maker_price_improvement(row):
    if row["makerOrderDirection"] == "long":
        return (row["oraclePrice"] / row["execPrice"]) - 1
    elif row["makerOrderDirection"] == "short":
        return (row["execPrice"] / row["oraclePrice"]) - 1

def calc_taker_price_improvement(row):
    if row["takerOrderDirection"] == "long":
        return (row["oraclePrice"] / row["execPrice"]) - 1
    elif row["takerOrderDirection"] == "short":
        return (row["execPrice"] / row["oraclePrice"]) - 1

def calc_color(row):
    if row['priceImprovement'] is None:
        return "white"
    elif float(row["priceImprovement"]) > 0:
        return "green"
    else:
        return "red"

def calculate_agg_counterparty_volume(fills_df: pd.DataFrame, user_make_or_taker: str):
    '''
    Returns a dataframe with the counterparty of each trade, and total base asset traded
    '''
    if user_make_or_taker == "maker":
        counterparty = "taker"
    else:
        counterparty = "maker"

    # only keep the top 10 rows sorted by base asset traded, group remainder into "other"

    agg_df = fills_df.groupby([counterparty]).agg(
        {
            "baseAssetAmountFilled": "sum",
            "quoteAssetAmountFilled": "sum",
        }
    )
    agg_df[counterparty] = agg_df.index
    agg_df[counterparty] = agg_df[counterparty].fillna("vAMM")
    agg_df.reset_index(drop=True, inplace=True)
    top_10 = agg_df.sort_values(by="baseAssetAmountFilled", ascending=False).head(10)
    other = agg_df.sort_values(by="baseAssetAmountFilled", ascending=False).tail(-10)
    other = pd.DataFrame(
        {
            "baseAssetAmountFilled": [other["baseAssetAmountFilled"].sum()],
            "quoteAssetAmountFilled": [other["quoteAssetAmountFilled"].sum()],
            counterparty: ["other"],
        },
        index=["other"],
    )
    base_total = agg_df["baseAssetAmountFilled"].sum()
    quote_total = agg_df["quoteAssetAmountFilled"].sum()
    res_df = pd.concat([top_10, other])
    res_df["basePercentage"] = res_df['baseAssetAmountFilled'] / base_total * 100.0
    res_df["quotePercentage"] = res_df['quoteAssetAmountFilled'] / quote_total * 100.0
    return res_df


async def show_user_volume(clearing_house: DriftClient):
    url = "https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/"
    url += "market/%s/trades/%s/%s/%s"
    mol1, molselect, mol0, mol2, _ = st.columns([3, 3, 3, 3, 10])
    market_name = mol1.selectbox(
        "market",
        ['All', 'All PERP', 'All SPOT'] + ALL_MARKET_NAMES,
        index=3
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
    if market_name == 'All':
        market_names = ALL_MARKET_NAMES
    elif market_name == 'All PERP':
        market_names = [x for x in ALL_MARKET_NAMES if '-PERP' in x]
    elif market_name == 'All SPOT':
        market_names = [x for x in ALL_MARKET_NAMES if '-PERP' not in x]
    else:
        market_names = [market_name]

    dfs, data_urls = await load_volumes_async(dates, market_names, with_urls=True)
    dd, _ = st.columns([6, 5])
    with dd.expander(f"data sources ({len(data_urls)})"):
        st.write(data_urls)

    subset_dedup = list(set(dfs.columns) - set(['recordKey']))
    dfs = dfs.drop_duplicates(subset=subset_dedup)
    selected = st.radio('', ["Volume", "Referrals"], index=0, horizontal=True)

    tabs = st.tabs(['overview', 'user breakdown', 'fill quality', 'lp breakdown', 'score', 'snapshot inspect'])
    
    df = pd.DataFrame()
    if selected == 'Volume':
        'orderFilledWithAmm'
        mapping_dict = {'orderFilledWithAmm': 'vAMM',
                        'orderFillWithPhoenix': 'Phoenix',
                        'orderFillWithSerum': 'Openbook',
                        'orderFilledWithLpJit': 'vAMM',
                        'orderFilledWithAmmJitLpSplit': 'vAMM',
                        'orderFilledWithAmmJit': 'vAMM',
                        }

        # Apply the mapping using map and provide a default value of 'external' for values not found in the dictionary
        # dfs['taker'] = dfs['actionExplanation'].map(mapping_dict).fillna('external')
        dfs.loc[dfs['taker'] == 'undefined', 'taker'] = dfs.loc[dfs['taker'] == 'undefined', 'actionExplanation'].map(mapping_dict).fillna('external')
        dfs.loc[dfs['maker'] == 'undefined', 'maker'] = dfs.loc[dfs['maker'] == 'undefined', 'actionExplanation'].map(mapping_dict).fillna('external')

        # dfs['taker'] = dfs['taker'].replace('undefined', 'vAMM' if 'perp' in market_name.lower() else 'external')
        # dfs['maker'] = dfs['maker'].replace('undefined', 'vAMM' if 'perp' in market_name.lower() else 'external')
        maker_volume = dfs.groupby("maker")["quoteAssetAmountFilled"].sum().fillna(0)
        taker_volume = dfs.groupby("taker")["quoteAssetAmountFilled"].sum().fillna(0)
        maker_jit_volume = dfs[dfs['actionExplanation']=='orderFilledWithMatchJit'].groupby("maker")["quoteAssetAmountFilled"].sum().fillna(0)

        df = pd.concat(
            {"maker volume": maker_volume, 
             "taker volume": taker_volume,
             "maker jit volume": maker_jit_volume,
             }, axis=1
        )
        df['total volume'] = df[['maker volume', 'taker volume']].sum(axis=1)
        df = df[['total volume', 'taker volume', 'maker volume', 'maker jit volume']]
        df = df.sort_values("total volume", ascending=False)

    with tabs[0]:
        # st.write(dfs.columns)
        s1,s2,s3 = st.columns(3)
        s1.metric('total volume:', f'${dfs["quoteAssetAmountFilled"].sum():,.2f}', f'{len(dfs):,} trades')
        s2.metric('total fees:', f'${dfs[["takerFee"]].sum().sum():,.2f} taker',
                  f'{len(dfs["taker"].unique())} unique takers',
                  )
        s3.metric('total rebates:', f'${dfs[["makerFee"]].sum().sum():,.2f} maker',
                  f'${-dfs[["fillerReward"]].sum().sum():,.2f} filler',
                  )
        fill_types = dfs.groupby('actionExplanation')['quoteAssetAmountFilled'].sum()\
            .sort_values(ascending=False)
        st.write(fill_types)

    with tabs[1]:
        if selected == 'Volume':
            s1,s2,s3 = st.columns(3)
            s1.metric('total volume:', f'${df["taker volume"].sum():,.2f}', f'{len(df):,} unique traders')
            st.dataframe(df, use_container_width=True)

        if selected == 'Referrals':
            opt = st.radio("users:", ["by trader", "by referrer"])
            if "referrerReward" in dfs.columns:
                reward_volume = pd.DataFrame(
                    dfs[dfs.referrerReward.replace('undefined', 0).fillna(0).astype(float) > 0]
                    .groupby("taker")["quoteAssetAmountFilled"]
                    .sum()
                )
                reward_volume.columns = ["taker volume"]
                if opt == "by referrer":
                    user_stats_df = await get_user_stats(clearing_house)
                    user_stats_df = user_stats_df[
                        user_stats_df.referrer.astype(str)
                        != "11111111111111111111111111111111"
                    ]

                    def get_subaccounts(authority, number_of_sub_accounts_created):
                        userAccountKeys = []
                        for sub_id in range(number_of_sub_accounts_created):
                            user_account_pk = get_user_account_public_key(
                                clearing_house.program_id, authority, sub_id
                            )
                            userAccountKeys.append(str(user_account_pk))
                        return ",".join(userAccountKeys)

                    user_stats_df["subAccounts"] = user_stats_df.apply(
                        lambda x: get_subaccounts(
                            x["authority"], x["number_of_sub_accounts_created"]
                        ),
                        axis=1,
                    )
                    reward_volume = reward_volume.reset_index()
                    reward_volume['taker'] = reward_volume['taker'].astype(str)
                    reward_volume.index = reward_volume['taker'].apply(lambda x: str(user_stats_df[user_stats_df['subAccounts'].str.contains(x)]['referrer'].dropna().values[0]))
                    reward_volume.index.name = 'referrer'
                    reward_volume.columns = ['taker', 'referred taker volume']
                    reward_volume = reward_volume.reset_index().groupby('referrer').sum()
                    
                st.dataframe(reward_volume)

                if opt == 'by referrer':
                    with st.expander('referrer map'):
                        st.dataframe(user_stats_df[["authority", "subAccounts", "referrer"]])
            else:
                st.code("no referrerRewards were seen in this market")

    


    with tabs[2]:
        if selected == 'Volume':

            n_bins = st.number_input("Fill Bins:", value=100)
            authority = st.text_input(
                "Public Key:",
                value=dfs['maker'].values[0],
            )
            auth_maker_df = dfs[dfs["maker"] == authority]
            auth_taker_df = dfs[dfs["taker"] == authority]

            st.text(
                "Price improvement is defined as `oraclePrice/execPrice - 1` if pubKey is long, and `execPrice/oraclePrice - 1` if pubKey is short."
            )
            start_date = dt.fromtimestamp(auth_maker_df["ts"].min())
            end_date = dt.fromtimestamp(auth_maker_df["ts"].max())
            st.text(f"Start date: {start_date} -> end date: {end_date}")

            if auth_maker_df.shape[0] > 0 or auth_taker_df.shape[0] > 0:
                color_discrete_map = {"green": "rgb(255,0,50)", "green": "rgb(0,255,50)"}


                st.markdown("## Maker fills:")
                if not auth_maker_df.empty:
                    col11, col12 = st.columns(2)
                    auth_maker_df["execPrice"] = auth_maker_df.apply(calc_maker_exec_price, axis=1)
                    auth_maker_df["ts"] = pd.to_datetime(auth_maker_df["ts"], unit="s", utc=True)
                    auth_maker_df["priceImprovement"] = auth_maker_df.apply(
                        calc_maker_price_improvement, axis=1
                    )
                    auth_maker_df["color"] = auth_maker_df.apply(calc_color, axis=1)

                    st.dataframe(
                        pd.DataFrame(
                            {
                                "ts": auth_maker_df["ts"],
                                "slot": auth_maker_df["slot"],
                                "maker": auth_maker_df["maker"],
                                "makerOrderId": auth_maker_df["makerOrderId"],
                                "makerOrderDirection": auth_maker_df["makerOrderDirection"],
                                "taker": auth_maker_df["taker"],
                                "takerOrderId": auth_maker_df["takerOrderId"],
                                "takerOrderDirection": auth_maker_df["takerOrderDirection"],
                                "baseAssetAmountFilled": auth_maker_df["baseAssetAmountFilled"],
                                "quoteAssetAmountFilled": auth_maker_df["quoteAssetAmountFilled"],
                                "execPrice": auth_maker_df["execPrice"],
                                "oraclePrice": auth_maker_df["oraclePrice"],
                                "priceImprovement": auth_maker_df["priceImprovement"],
                                "actionExplanation": auth_maker_df["actionExplanation"],
                                "txSig": auth_maker_df["txSig"],
                                "color": auth_maker_df["color"],
                            }
                        )
                    )

                    fig = px.histogram(
                        auth_maker_df,
                        x=auth_maker_df["priceImprovement"],
                        y=auth_maker_df["baseAssetAmountFilled"],
                        nbins=n_bins,
                        labels={"x": "priceImprovement", "y": "baseAssetAmountFilled"},
                        color="color",
                        color_discrete_map=color_discrete_map,
                        opacity=0.3,
                    )
                    fig.add_vline(x=0, line_color="red", annotation_text="OraclePrice")
                    fig = fig.update_layout(
                        xaxis_title="Price Improvement on Maker Fills (positive is better)",
                        yaxis_title="Base amount filled at this price improvement",
                    )
                    col11.plotly_chart(fig)

                    fig = px.scatter(
                        auth_maker_df,
                        x=auth_maker_df["ts"],
                        y=auth_maker_df["priceImprovement"],
                        labels={"x": "ts", "y": "priceImprovement"},
                        color="color",
                        color_discrete_map=color_discrete_map,
                        opacity=0.3,
                    )
                    fig = fig.update_layout(
                        xaxis_title="Price Improvement over time",
                        yaxis_title="Price improvement (positive is better))",
                    )
                    col12.plotly_chart(fig)

                    st.markdown("### Counterparty fills:")
                    col21, col22 = st.columns(2)
                    counterparty_df = calculate_agg_counterparty_volume(auth_maker_df, 'maker')

                    fig = px.pie(counterparty_df, values='baseAssetAmountFilled', title='Counterparty Base Volume', names='taker')
                    col21.plotly_chart(fig)

                    fig = px.pie(counterparty_df, values='quoteAssetAmountFilled', title='Counterparty Quote Volume', names='taker')
                    col22.plotly_chart(fig)

                    st.dataframe(
                        pd.DataFrame(
                            {
                                "taker": counterparty_df["taker"],
                                "baseAssetAmountFilled": counterparty_df["baseAssetAmountFilled"],
                                "quoteAssetAmountFilled": counterparty_df["quoteAssetAmountFilled"],
                                "basePercentage": counterparty_df['basePercentage'],
                                "quotePercentage": counterparty_df['quotePercentage'],
                            }
                        )
                    )
                else:
                    st.write("No maker fills")

                st.markdown("## Taker fills:")
                if not auth_taker_df.empty:
                    col11, col12 = st.columns(2)
                    auth_taker_df["execPrice"] = auth_taker_df.apply(calc_taker_exec_price, axis=1)
                    auth_taker_df["ts"] = pd.to_datetime(auth_taker_df["ts"], unit="s", utc=True)
                    auth_taker_df["priceImprovement"] = auth_taker_df.apply(
                        calc_taker_price_improvement, axis=1
                    )
                    auth_taker_df["color"] = auth_taker_df.apply(calc_color, axis=1)

                    st.dataframe(
                        pd.DataFrame(
                            {
                                "ts": auth_taker_df["ts"],
                                "slot": auth_taker_df["slot"],
                                "maker": auth_taker_df["maker"],
                                "makerOrderId": auth_taker_df["makerOrderId"],
                                "makerOrderDirection": auth_taker_df["makerOrderDirection"],
                                "taker": auth_taker_df["taker"],
                                "takerOrderId": auth_taker_df["takerOrderId"],
                                "takerOrderDirection": auth_taker_df["takerOrderDirection"],
                                "baseAssetAmountFilled": auth_taker_df["baseAssetAmountFilled"],
                                "quoteAssetAmountFilled": auth_taker_df["quoteAssetAmountFilled"],
                                "execPrice": auth_taker_df["execPrice"],
                                "oraclePrice": auth_taker_df["oraclePrice"],
                                "priceImprovement": auth_taker_df["priceImprovement"],
                                "actionExplanation": auth_taker_df["actionExplanation"],
                                "txSig": auth_taker_df["txSig"],
                                "color": auth_taker_df["color"],
                            }
                        )
                    )
                    fig = px.histogram(
                        auth_taker_df,
                        x="priceImprovement",
                        y="baseAssetAmountFilled",
                        nbins=n_bins,
                        labels={"x": "priceImprovement", "y": "baseAssetAmountFilled"},
                        color="color",
                        color_discrete_map=color_discrete_map,
                        opacity=0.3,
                    )
                    fig.add_vline(x=0, line_color="red", annotation_text="OraclePrice")
                    fig = fig.update_layout(
                        xaxis_title="Price Improvement on Taker Fills (positive is better)",
                        yaxis_title="Base amount filled at this price improvement",
                    )
                    col11.plotly_chart(fig)

                    fig = px.scatter(
                        auth_taker_df,
                        x=auth_taker_df["ts"],
                        y=auth_taker_df["priceImprovement"],
                        labels={"x": "ts", "y": "priceImprovement"},
                        color="color",
                        color_discrete_map=color_discrete_map,
                        opacity=0.3,
                    )
                    fig = fig.update_layout(
                        xaxis_title="Price Improvement over time",
                        yaxis_title="Price improvement (positive is better))",
                    )
                    col12.plotly_chart(fig)

                    st.markdown("### Counterparty fills:")
                    col21, col22 = st.columns(2)
                    counterparty_df = calculate_agg_counterparty_volume(auth_maker_df, 'taker')

                    fig = px.pie(counterparty_df, values='baseAssetAmountFilled', title='Counterparty Base Volume', names='maker')
                    col21.plotly_chart(fig)

                    fig = px.pie(counterparty_df, values='quoteAssetAmountFilled', title='Counterparty Quote Volume', names='maker')
                    col22.plotly_chart(fig)

                    st.dataframe(
                        pd.DataFrame(
                            {
                                "maker": counterparty_df["maker"],
                                "baseAssetAmountFilled": counterparty_df["baseAssetAmountFilled"],
                                "quoteAssetAmountFilled": counterparty_df["quoteAssetAmountFilled"],
                                "basePercentage": counterparty_df['basePercentage'],
                                "quotePercentage": counterparty_df['quotePercentage'],
                            }
                        )
                    )
                else:
                    st.write("No taker fills")

            else:
                st.markdown(f"public key not found...")

    with tabs[4]:
        # total score per week is 0.00125
        dfs2 = dfs

        dfs2['volumeScore1'] = dfs2['quoteAssetAmountFilled'] * \
            dfs2['actionExplanation'].apply(lambda x: {'orderFilledWithMatchJit': 10}.get(x, 1))
        market_groupings = dfs.groupby(['marketType', 'marketIndex'])[['quoteAssetAmountFilled', 'volumeScore1']].sum()\
            .sort_values(by='quoteAssetAmountFilled', ascending=False)

        market_groupings = pd.DataFrame(market_groupings)

        market_groupings['percentOfVolume'] = market_groupings['quoteAssetAmountFilled']/market_groupings['quoteAssetAmountFilled'].sum() * 100
        market_groupings['ScorePercentage'] = market_groupings['percentOfVolume']

        # Normalize to sum to 100 and update ScorePercentage
        total_percentage = market_groupings['percentOfVolume'].sum()
        market_groupings['ScorePercentage'] = market_groupings['ScorePercentage'] / total_percentage * 100


        # Define lower bounds
        perp_lower_bound = 0.5
        other_lower_bound = 0.05
        # Apply lower bounds based on marketType
        mask_perp = market_groupings.index.get_level_values('marketType').str.contains('perp')
        market_groupings.loc[mask_perp, 'ScorePercentage'] = market_groupings.loc[mask_perp, 'ScorePercentage'].clip(
            lower=perp_lower_bound
        )

        market_groupings.loc[~mask_perp, 'ScorePercentage'] = market_groupings.loc[
            ~mask_perp, 'ScorePercentage'].clip(lower=other_lower_bound)

        # Adjust the remaining values to ensure the total sums to 100
        non_lb_markets = market_groupings['ScorePercentage'] == market_groupings['percentOfVolume']
        extra_amounts = 1 - (100 - market_groupings[non_lb_markets]['ScorePercentage'] .sum())/100

        # Subtract the extra amounts from rows that weren't clipped with a lower bound
        market_groupings.loc[non_lb_markets, 'ScorePercentage'] = market_groupings.loc[non_lb_markets, 'ScorePercentage'] * extra_amounts
        # Display the result
        st.write(market_groupings)

        # res = dfs2.groupby(['marketType', 'marketIndex', 'maker'])['volumeScore1']
        # Define constant value N
        N = st.number_input('score given away each weekly period:', 0.005/4 * 100)

        st.write('75/25 pct split between volume/liquidity score')

        # Multiply each row in dfs2 by the applicable ScorePercentage/100 * N * 0.75 * 1/volumeScore1
        dfs2['multiplied_score'] = dfs2.apply(lambda row: row['volumeScore1'] * \
                                              (market_groupings.loc[(row['marketType'], row['marketIndex']), 'ScorePercentage'] / 100) \
                                                * N * 0.75 
                                                * (1 / market_groupings.loc[(row['marketType'], row['marketIndex']), 'volumeScore1']), axis=1)

        # Display the multiplied_score column in dfs2
        st.metric('total maker volume score distributed', dfs2['multiplied_score'].sum())
        
        st.header('user scores')
        st.write(dfs2.groupby(['maker'])['multiplied_score'].sum().sort_values(ascending=False))

        st.header('user scores by market')
        st.write(dfs2.groupby(['marketType', 'marketIndex', 'maker'])['multiplied_score'].sum().sort_values(ascending=False))

                
    lp_result = None
    with tabs[3]:
        dolpeval = st.radio('do lp eval:', [True, False], index=1)
        if dolpeval:
            lp_dfs, data_urls = await load_lp_info_async(dates, market_names, True)
            with st.expander(f'{len(data_urls)} sources:'):
                st.write(data_urls)
            lp_result = lp_dfs
            lp_result['marketIndex'] = lp_result['name'].astype(int)
            lp_agg_result = lp_dfs[['name', 'slot', 'sqrtK', 'lpShares']].astype(int).groupby(['name', 'slot']).agg({'sqrtK':'mean', 'lpShares':'sum'})
            # lp_result.to_csv("~/tmp2.csv")

            st.write(lp_agg_result)
            
             
        #     import requests
        #     url = f'https://dlob-data.s3.eu-west-1.amazonaws.com/mainnet-beta/{dates[0].strftime("%Y-%m-%d")}/lp-shares-index.json.gz'
        #     try:
        #         index_lp = requests.get(url).json()
        #         st.write(index_lp)
        #     except:
        #         st.warning('cannot load ' + url)
        #     from driftpy.constants.perp_markets import mainnet_perp_market_configs

        #     market_index = [x.market_index for x in mainnet_perp_market_configs if x.symbol == market_names[0]][0]
        #     url2 = f'https://dlob-data.s3.eu-west-1.amazonaws.com/mainnet-beta/{dates[0].strftime("%Y-%m-%d")}/lp-shares-{market_index}.csv.gz'
        #     st.write(url2)
        #     lps_over_time = pd.read_csv(url2)
        #     st.write(lps_over_time)

        #     lp_over_time_df = lps_over_time[['slot', 'sqrtK', 'lpShares']].astype(int).groupby('slot').agg({'sqrtK':'mean', 'lpShares':'sum'})

        #     st.write(lp_over_time_df)
        #     st.plotly_chart(lp_over_time_df.plot())

    with tabs[4]:
        if lp_result is not None:
            st.header('dlp user scores by market')
            rrrr = dfs2[dfs2.maker=='vAMM'][['slot', 'marketType', 'marketIndex', 'maker', 'multiplied_score']]

            rrrr['marketIndex'] = rrrr['marketIndex'].astype(int)

            # t1 = pd.read_csv("~/tmp1.csv", index_col=[0]).sort_values('slot')
            # t2 = pd.read_csv("~/tmp2.csv", index_col=[0]).sort_values('slot')

            st.write(rrrr)
            st.write(lp_result)
            merged_df = pd.merge_asof(lp_result.sort_values('slot'), 
                                      rrrr.sort_values('slot'), 
                                      by='marketIndex', on='slot', direction='forward')
            st.write(merged_df.shape)

            st.write(merged_df['multiplied_score'].sum(), rrrr['multiplied_score'].sum())

            merged_df['dlp_multiplied_score'] = (merged_df['lpShares']/merged_df['sqrtK'])*merged_df['multiplied_score']
            dlp_scored = merged_df.groupby(['user', 
                                            # 'marketIndex', 
                                            'marketType']).agg({'dlp_multiplied_score':'sum'})
            
            st.write('total dlp score:', dlp_scored['dlp_multiplied_score'].sum())
            st.write(dlp_scored.shape)
            st.write(dlp_scored)

            dlp_scored_by_market = merged_df.groupby([
                # 'user', 
                                            'marketIndex', 
                                            'marketType']).agg({'dlp_multiplied_score':'sum'})
            
            st.write('total dlp score by market:', dlp_scored_by_market['dlp_multiplied_score'].sum())
            st.write(dlp_scored_by_market.shape)
            st.write(dlp_scored_by_market)

            # st.write(rrrr)
            # rrrr.to_csv("~/tmp1.csv")


            # res2 = rrrr.merge(lp_result)
            # st.write(res2)

    with tabs[5]:
        vvv = st.text_input('source:',

            value='https://dlob-data.s3.eu-west-1.amazonaws.com/mainnet-beta/2024-01-11/241100306.json.gz'
        )
        dolpeval = st.radio('load raw snap:', [True, False], index=1)
        if dolpeval:
            import requests
            r = requests.get(vvv).json()
            st.json(r)