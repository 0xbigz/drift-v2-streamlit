
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
import os
import json
import streamlit as st
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio

async def show_user_stats(url: str, clearing_house: ClearingHouse):
    ch = clearing_house

    all_user_stats = await ch.program.account['UserStats'].all()
    kp = Keypair()
    ch = ClearingHouse(ch.program, kp)

    df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    fees_df = pd.DataFrame([x.account.fees.__dict__ for x in all_user_stats])


    # print(df.columns)
    print(fees_df.columns)
    df = pd.concat([df, fees_df], axis=1)

    for x in df.columns:
        if x in ['taker_volume30d', 'maker_volume30d', 'filler_volume30d', 'total_fee_paid', 'total_fee_rebate']:
            df[x] /= 1e6
    df = df[['authority', 'taker_volume30d', 'maker_volume30d', 'filler_volume30d', 'total_fee_paid', 'total_fee_rebate', 'number_of_sub_accounts']]

    st.dataframe(df)
        
    