from datetime import datetime as dt
import sys
from tokenize import tabsize
import driftpy
import pandas as pd
import numpy as np
from driftpy.math.oracle import *

import plotly.express as px

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_user_account,
    get_state_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.addresses import *
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime


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


async def get_user_stats(clearing_house: ClearingHouse):
    ch = clearing_house
    all_user_stats = await ch.program.account["UserStats"].all()
    user_stats_df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    return user_stats_df


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

async def show_user_volume(clearing_house: ClearingHouse):
    url = "https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/"
    url += "market/%s/trades/%s/%s/%s"
    mol1, molselect, mol0, mol2, _ = st.columns([3, 3, 3, 3, 10])
    market_name = mol1.selectbox(
        "market",
        ["SOL-PERP", "BTC-PERP", "ETH-PERP", "1MBONK-PERP", "MATIC-PERP", "ARB-PERP", "DOGE-PERP", "SOL"],
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

    dfs, data_urls = load_volumes(dates, market_name, with_urls=True)
    dd, _ = st.columns([6, 5])
    with dd.expander("data sources"):
        st.write(data_urls)

    selected = st.radio('', ["Volume", "Referrals"], index=0, horizontal=True)
    if selected == 'Volume':
        maker_volume = dfs.groupby("maker")["quoteAssetAmountFilled"].sum()
        taker_volume = dfs.groupby("taker")["quoteAssetAmountFilled"].sum()
        df = pd.concat(
            {"maker volume": maker_volume, "taker volume": taker_volume}, axis=1
        )
        df = df.sort_values("maker volume", ascending=False)
        st.dataframe(df)

        n_bins = st.number_input("Fill Bins:", value=100)
        authority = st.text_input(
            "Public Key:",
            value=df.index[0],
        )
        auth_maker_df = dfs[dfs["maker"] == authority]
        auth_taker_df = dfs[dfs["taker"] == authority]

        st.text(
            "Price improvement is defined as 'execPrice/oraclePrice - 1' if pubKey is long, and ' oraclePrice/execPrice - 1' if pubKey is short."
        )
        start_date = dt.fromtimestamp(auth_maker_df["ts"].min())
        end_date = dt.fromtimestamp(auth_maker_df["ts"].max())
        st.text(f"Start date: {start_date} -> end date: {end_date}")

        if auth_maker_df.shape[0] > 0 or auth_taker_df.shape[0] > 0:

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
                # return row["quoteAssetAmountFilled"] / row["baseAssetAmountFilled"]

            def calc_taker_exec_price(row):
                if row["takerOrderDirection"] == "long":
                    # user is long, paying more is bad, add fee to quoteAmt (negative fee is rebate)
                    return (row["quoteAssetAmountFilled"] + row["makerFee"]) / row[
                        "baseAssetAmountFilled"
                    ]
                elif row["takerOrderDirection"] == "short":
                    # user is short, recv more is good, sub fee from quoteAmt (negative fee is rebate)
                    return (row["quoteAssetAmountFilled"] - row["makerFee"]) / row[
                        "baseAssetAmountFilled"
                    ]
                # return row["quoteAssetAmountFilled"] / row["baseAssetAmountFilled"]

            def calc_maker_price_improvement(row):
                if row["makerOrderDirection"] == "long":
                    return (row["execPrice"] / row["oraclePrice"]) - 1
                elif row["makerOrderDirection"] == "short":
                    return (row["oraclePrice"] / row["execPrice"]) - 1

            def calc_taker_price_improvement(row):
                if row["takerOrderDirection"] == "long":
                    return (row["execPrice"] / row["oraclePrice"]) - 1
                elif row["takerOrderDirection"] == "short":
                    return (row["oraclePrice"] / row["execPrice"]) - 1

            def calc_color(row):
                if float(row["priceImprovement"]) > 0:
                    return "green"
                else:
                    return "red"

            color_discrete_map = {"green": "rgb(255,0,50)", "green": "rgb(0,255,50)"}

            auth_maker_df["execPrice"] = auth_maker_df.apply(calc_maker_exec_price, axis=1)
            auth_taker_df["execPrice"] = auth_taker_df.apply(calc_taker_exec_price, axis=1)

            st.markdown("## Maker fills:")
            col1, col2 = st.columns(2)
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
                        "taker": auth_maker_df["taker"],
                        "execPrice": auth_maker_df["execPrice"],
                        "oraclePrice": auth_maker_df["oraclePrice"],
                        "priceImprovement": auth_maker_df["priceImprovement"],
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
            col1.plotly_chart(fig)

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
            col2.plotly_chart(fig)

            st.markdown("## Taker fills:")
            col1, col2 = st.columns(2)
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
                        "taker": auth_taker_df["taker"],
                        "execPrice": auth_taker_df["execPrice"],
                        "oraclePrice": auth_taker_df["oraclePrice"],
                        "priceImprovement": auth_taker_df["priceImprovement"],
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
            col1.plotly_chart(fig)

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
            col2.plotly_chart(fig)

        else:
            st.markdown(f"public key not found...")
    if selected == 'Referrals':
        opt = st.radio("users:", ["by trader", "by referrer"])
        if "referrerReward" in dfs.columns:
            reward_volume = pd.DataFrame(
                dfs[dfs.referrerReward > 0]
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

    
