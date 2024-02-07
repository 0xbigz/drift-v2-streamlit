from datetime import datetime as dt
import sys
from tokenize import tabsize
import driftpy
import pandas as pd
import numpy as np
from driftpy.accounts.oracle import *
from constants import ALL_MARKET_NAMES

import plotly.express as px
from datafetch.s3_fetch import load_volumes

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
from driftpy.constants.spot_markets import mainnet_spot_market_configs, devnet_spot_market_configs, SpotMarketConfig
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


                
@st.cache(ttl=1800)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

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
        if response.status_code == 403 or response.status_code == 404:
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

    if len(dfs):
        result_df = pd.concat(dfs)

        if with_urls:
            return result_df, data_urls

        return result_df

async def fetch_volume_data(date, market_name):
    if market_name == 'JitoSOL':
        market_name = 'jitoSOL'

    month_str = ('0'+str(date.month))[-2:]
    day_str = ('0'+str(date.day))[-2:]

    # url = f"https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/{market_name}/trades/{date.year}/{date.month}/{date.day}"
    prefix = 'https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH'
    url = f"{prefix}/market/{market_name}/tradeRecords/{date.year}/{date.year}{month_str}{day_str}"
    # st.write(url)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 403:
            st.write(url)
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
    if len(dfs):
        result_df = pd.concat(dfs)

        if with_urls:
            return result_df, data_urls

        return result_df
    return pd.DataFrame(), []


async def fetch_liquidity_data(market_name):
    if market_name == 'JitoSOL':
        market_name = 'jitoSOL'

    if '-perp' in market_name.lower():
        market_index = [x.market_index for x in mainnet_perp_market_configs if market_name == x.symbol]
    else:
        market_index = [x.market_index for x in mainnet_spot_market_configs if market_name == x.symbol]
  
    if len(market_index) == 1:
        market_index = market_index[0]
    else:
        st.warning('failed to find index for '+ market_name)
        return None, ''
    
    market_type = 'spot'
    if 'perp' in market_name.lower():
        market_type = 'perp'

    url = f"https://dlob-data.s3.eu-west-1.amazonaws.com/mainnet-beta/aggregate-liq-score/{market_type}-{str(market_index)}.csv.gz"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 403:
            st.warning('failed to get '+ market_name + ' '+ url)
            return None, ''
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df['market_name'] = market_name
        df['market_index'] = market_index
        df['market_type'] = market_type
        return df, url

async def load_liquidity_agg_score_async(market_names, with_urls=False):
    # doesnt use date
    data_urls = []
    dfs = []

    tasks = [fetch_liquidity_data(market_name) for market_name in market_names]

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

    if len(dfs) == 0:
        st.warning('historical data unavailable for this date. ' + str(data_urls))
        return 0

    subset_dedup = list(set(dfs.columns) - set(['recordKey']))
    dfs = dfs.drop_duplicates(subset=subset_dedup)
    try:
        dfs['maker'] = dfs['maker'].fillna('vAMM')
        dfs['taker'] = dfs['taker'].fillna('vAMM')
    except:
        pass

    selected = st.radio('', ["Volume", "Referrals"], index=0, horizontal=True)

    tabs = st.tabs(['overview', 'user breakdown', 'fill quality', 'score', 'snapshot inspect'])
    
    volume_df = pd.DataFrame()
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

        volume_df = pd.concat(
            {"maker volume": maker_volume, 
             "taker volume": taker_volume,
             "maker jit volume": maker_jit_volume,
             }, axis=1
        )
        volume_df['total volume'] = volume_df[['maker volume', 'taker volume']].sum(axis=1)
        volume_df = volume_df[['total volume', 'taker volume', 'maker volume', 'maker jit volume']]
        volume_df = volume_df.sort_values("total volume", ascending=False)

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
            s1.metric('total volume:', f'${volume_df["taker volume"].sum():,.2f}',
                       f'{len(volume_df):,} unique traders')
            st.dataframe(volume_df, use_container_width=True)


            csv = convert_df(volume_df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='user_volume_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv',
                mime='text/csv',
            )


            st.plotly_chart(volume_df['total volume']\
                            .reset_index(drop=True)\
                                .sort_values().plot(log_y=True))
            a1, a2 = st.columns(2)
            a_min = a1.number_input('volume', 1000)
            a = a2.number_input('volume', 20000)

            st.write(volume_df[(volume_df['total volume']>=a_min) & (volume_df['total volume']<=a)]['taker volume'].sum(), 
                     volume_df[volume_df['total volume']<=a].sum(), 
                     volume_df[volume_df['total volume']>=a].sum(), 
                     len(volume_df[volume_df['total volume']>=a]), 
                     len(volume_df)
                     )

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

    min_slot_in_view = 0
    lp_result = None

    with tabs[3]:
        # total score per week is 0.00125
        s1, s2 = st.columns(2)
        N = s1.number_input('score given away each weekly period:', 
                            2e6
                            # 0.005/4 * 100
                            )
        s1.write('37.5-62.5 taker/maker. maker: 70/30 pct split between volume/liquidity score')

        N_M = N * 0.625
        N_T = N * 0.375

        dolpeval = s2.radio('do lp eval:', [True, False], index=1)

        subtabs = st.tabs(['overview', 'per market alloc', 'volume only', 'liquidity only'])
        dfs2 = dfs

        def get_ticket_multiplier(row):
            # st.write(row)
            if row['marketIndex']==24 and row['marketType']=='perp':
                return 2
            if row['marketIndex']==11 and row['marketType']=='spot':
                return 2
            return 1


        dfs2['volumeScore1'] = dfs2['quoteAssetAmountFilled'] * \
            dfs2['actionExplanation'].apply(lambda x: {'orderFilledWithMatchJit': 10}.get(x, 1))
        dfs2['ticketScore1'] = dfs2['takerFee'] * dfs2.apply(get_ticket_multiplier, axis=1)        
        market_groupings = dfs.groupby(['marketType', 'marketIndex'])[['quoteAssetAmountFilled', 'volumeScore1']].sum()\
            .sort_values(by='quoteAssetAmountFilled', ascending=False)

        min_slot_in_view = dfs2.slot.min()

        market_groupings = pd.DataFrame(market_groupings)

        market_groupings['percentOfVolume'] = market_groupings['quoteAssetAmountFilled']/market_groupings['quoteAssetAmountFilled'].sum() * 100
        market_groupings['ScorePercentage'] = market_groupings['percentOfVolume']

        # Normalize to sum to 100 and update ScorePercentage
        total_percentage = market_groupings['percentOfVolume'].sum()
        market_groupings['ScorePercentage'] = market_groupings['ScorePercentage'] / total_percentage * 100



        # Define lower bounds
        perp_lower_bound = 0.52
        other_lower_bound = 0.052
        # Apply lower bounds based on marketType
        mask_perp = market_groupings.index.get_level_values('marketType').str.contains('perp')
        market_groupings.loc[mask_perp, 'ScorePercentage'] = market_groupings.loc[mask_perp, 'ScorePercentage'].clip(
            lower=perp_lower_bound
        )
        market_groupings.loc[~mask_perp, 'ScorePercentage'] = market_groupings.loc[
            ~mask_perp, 'ScorePercentage'].clip(lower=other_lower_bound)
        attempts = 0
        while (market_groupings['ScorePercentage'].sum() - 100) > 1e-2 and attempts < 15:
            # st.write(attempts, market_groupings)
            # Adjust the remaining values to ensure the total sums to 100
            market_groupings.loc[mask_perp, 'ScorePercentage'] = market_groupings.loc[mask_perp, 'ScorePercentage'].clip(
            lower=perp_lower_bound
            )
            market_groupings.loc[~mask_perp, 'ScorePercentage'] = market_groupings.loc[
                ~mask_perp, 'ScorePercentage'].clip(lower=other_lower_bound)
            

            market_groupings['ScorePercentage'] /= (market_groupings['ScorePercentage'].sum() * .01)

            attempts +=1
            
        # Define constant value N
        with subtabs[1]:
            st.header('per market groupings')
            st.write(market_groupings)

            st.write(market_groupings.ScorePercentage.sum())

 

        # Multiply each row in dfs2 by the applicable ScorePercentage/100 * N * 0.7 * 1/volumeScore1
        dfs2['multiplied_score'] = dfs2.apply(lambda row: row['volumeScore1'] * \
                                              (market_groupings.loc[(row['marketType'], row['marketIndex']), 'ScorePercentage'] / 100) \
                                                * N_M * 0.7
                                                * (1 / market_groupings.loc[(row['marketType'], row['marketIndex']), 'volumeScore1']), axis=1)

        # Display the multiplied_score column in dfs2
        with subtabs[2]:
            st.metric('total maker volume score distributed', dfs2['multiplied_score'].sum(), 'vs '+ str(N*.7))
            
            st.header('user scores')
            st.write(dfs2.groupby(['maker'])['multiplied_score'].sum().sort_values(ascending=False))

            st.header('user scores by market')
            st.write(dfs2.groupby(['marketType', 'marketIndex', 'maker'])['multiplied_score'].sum().sort_values(ascending=False))
                    
            if dolpeval:
                lp_dfs, data_urls = await load_lp_info_async(dates, market_names, True)
                with st.expander(f'{len(data_urls)} sources:'):
                    st.write(data_urls)
                lp_result = lp_dfs
                lp_result['marketIndex'] = lp_result['name'].astype(int)
                lp_agg_result = lp_dfs[['name', 'slot', 'sqrtK', 'lpShares']].astype(int).groupby(['name', 'slot']).agg({'sqrtK':'mean', 'lpShares':'sum'})

                lp_agg_result['user own %'] = lp_agg_result['lpShares']/lp_agg_result['sqrtK'] * 100

                st.write(lp_agg_result)

            if lp_result is not None:
                st.header('dlp user scores by market')
                rrrr = dfs2[dfs2.maker=='vAMM'][['slot', 'marketType', 'marketIndex', 'maker', 'multiplied_score']]

                rrrr['marketIndex'] = rrrr['marketIndex'].astype(int)
                rrrr = rrrr.groupby(['slot', 'marketIndex', 'marketType', 'maker']).sum().reset_index()#.set_index('slot')
                
                unique_slots = lp_result.groupby('slot').sum().index
                def get_most_recent_slot(x):
                    for i in unique_slots:
                        if i >= x:
                            return i
                    return unique_slots[-1]
                        
                rrrr['lp_snap_slot'] = rrrr['slot'].apply(lambda x: get_most_recent_slot(x)).astype(int)
                rrrr = rrrr.groupby(['marketIndex', 'marketType', 'maker', 'lp_snap_slot']).sum().reset_index().drop(['slot'], axis=1)

                # t1 = pd.read_csv("~/tmp1.csv", index_col=[0]).sort_values('slot')
                # t2 = pd.read_csv("~/tmp2.csv", index_col=[0]).sort_values('slot')

                st.write(rrrr)
                merged_df = pd.merge_asof(lp_result.sort_values('slot'), 
                                        rrrr.sort_values('lp_snap_slot'), 
                                        by='marketIndex', left_on='slot', right_on='lp_snap_slot', direction='forward')
                st.write(merged_df.shape)
                st.write('too big to write')
                # st.write(merged_df)

                st.write(merged_df['multiplied_score'].sum(), rrrr['multiplied_score'].sum())

                merged_df['dlp_frac'] = (merged_df['lpShares']/merged_df['sqrtK']).apply(lambda x: min(.4, x))
                merged_df['dlp_multiplied_score'] = (merged_df['dlp_frac']*merged_df['multiplied_score'])

                # st.write(merged_df[merged_df.user==merged_df.user.unique()[0]])#.pivot_table(index='', column='user')
                #*merged_df['multiplied_score']
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

        with subtabs[3]:

            liquidity_dfs, liqudity_data_urls = await load_liquidity_agg_score_async(market_names, True)
            with st.expander(f'{len(liqudity_data_urls)} liquidity score sources:'):
                    st.write(liqudity_data_urls)
            st.write(liquidity_dfs.shape)
            # st.write(liquidity_dfs)

            # todo: filter by first slot of volume score 
            liquidity_dfs = liquidity_dfs[liquidity_dfs.slot > min_slot_in_view]
            st.write(f'after slot filter ({min_slot_in_view})',liquidity_dfs.shape)

            st.write(list(liquidity_dfs.columns))

            total_score_by_market = liquidity_dfs.groupby(['market_type', 'market_index'])['score'].sum()

            def get_mult_score(row):
                idx = (row['market_type'], row['market_index'])
                t1 = row['score'] / total_score_by_market.loc[idx]

                if idx in market_groupings.index:
                    t2 = t1 * (market_groupings.loc[idx, 'ScorePercentage'] / 100) * N_M * 0.3 
                    return t2
                else:
                    # not in market grouping
                    return 0
                

            ldf3 = liquidity_dfs.groupby(['user', 'market_type', 'market_index'])[['score']].sum().reset_index()
            ldf3['multiplied_score'] = ldf3.apply(lambda x: get_mult_score(x), axis=1)
            st.write(ldf3.shape)

            # st.write(ldf3.head(1000))

            st.write(f'ensure it all sums to target { N_M * 0.3 }', 'vs', f'{ldf3["multiplied_score"].sum()}')
            st.write(ldf3.groupby(['market_type', 'market_index']).sum())


        with subtabs[0]:
            if dolpeval:

                ums = pd.read_csv('https://gist.githubusercontent.com/0xbigz/be8c5205c41f51d1b131b41b114546ff/raw/c25811ee4f2c2999c72ff5f1e780e470fc4fb0e1/usermap_snapshot_20240201.csv',
                         index_col=[0]
                         )
                
                liq_comp = ldf3.groupby('user')[['multiplied_score']].sum()
                liq_comp.index = [x if x !='vamm' else 'vAMM' for x in liq_comp.index ]
                liq_comp.index.name = 'user'
                liq_comp.columns = ['liq_multiplied_score']
                liq_comp['marketType'] = 'perp'

                vol_comp_dlp = dlp_scored['dlp_multiplied_score']#.sort_values(ascending=False)
                # vol_comp_dlp.name = 'multiplied_score'
                vol_comp_dlp = vol_comp_dlp.reset_index().set_index('user')


                vol_comp = dfs2.groupby(['maker'])[['multiplied_score']].sum()#.sort_values(ascending=False)
                vol_comp.index.name = 'user'

                # remove dlp attributed volume score
                vol_comp.loc['vAMM', 'multiplied_score'] -= vol_comp_dlp['dlp_multiplied_score'].sum()


                dfs2['ticketScore1'] = dfs2['ticketScore1'].clip(0, 200_000)
                # st.write('N', N)
                taker_score = (dfs2.groupby(['taker'])['ticketScore1'].sum() / dfs2['ticketScore1'].sum()) * N_T
                taker_score.index.name = 'user'
                taker_score = pd.DataFrame(taker_score)
                taker_score.columns = ['taker_score']
                # taker_score.name = 'taker_score'

                # st.write(taker_score)

                # st.write(liq_comp)
                # st.write(vol_comp_dlp)
                # st.write(vol_comp)
                volume_cl = volume_df[['total volume']]
                volume_cl.index.name = 'user'
                volume_cl.columns = ['total_order_volume']
                # st.write(volume_df['total_volume'])
                fin = pd.concat([vol_comp, liq_comp, vol_comp_dlp, taker_score,
                volume_df['total volume']
                
                ]).fillna(0)
                st.write(fin)
                fin = fin.reset_index().groupby('user').sum()
                fin['maker_score'] = fin[['dlp_multiplied_score',
                                          'multiplied_score', 
                                          'liq_multiplied_score',
                                          ]].sum(axis=1)
                
                
                # fin = fin.groupby('user').fillna(0).sum()
                st.metric('maker_score score distributed', f'{fin["maker_score"].sum()}')
                
                
                fin = fin.reset_index()
                fin['user'] = fin['user'].astype(str)
                st.write(ums[['userAccount', 'authority']])
                fin = fin.merge(ums[['userAccount', 'authority']], left_on='user', right_on='userAccount')
                fin['total_score'] = fin['maker_score']+fin['taker_score']

                st.write('total user score:', fin['total_score'].sum())
                st.write(fin)

                csv = convert_df(fin)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='score_mock_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv',
                    mime='text/csv',
                )

                st.write(fin.groupby('authority')['total_score'].sum().sort_values())
            else:
                st.write('must turn on do lp eval')
    with tabs[4]:
        vvv = st.text_input('source:',

            value='https://dlob-data.s3.eu-west-1.amazonaws.com/mainnet-beta/2024-01-11/241100306.json.gz'
        )
        dolpeval = st.radio('load raw snap:', [True, False], index=1)
        if dolpeval:
            import requests
            r = requests.get(vvv).json()
            st.json(r)
        # st.write(dfs)