
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
from driftpy.math.margin import MarginCategory

async def show_user_health(clearing_house: ClearingHouse):

    user_authority = st.text_input('user authority', '')

    if user_authority != '':
        ch = clearing_house
        user_authority_pk = PublicKey(user_authority)

        chu = ClearingHouseUser(
            ch, 
            authority=user_authority_pk, 
            subaccount_id=0, 
            use_cache=True
        )
        await chu.set_cache()
        cache = chu.CACHE

        user_stats_pk = get_user_stats_account_public_key(ch.program_id, user_authority_pk)
        all_user_stats = await ch.program.account['UserStats'].fetch(user_stats_pk)
        print(all_user_stats)
        st.dataframe(pd.DataFrame(all_user_stats.__dict__).T[0])

        all_summarys = []
        for x in range(all_user_stats.number_of_sub_accounts_created):
            chu_sub = ClearingHouseUser(
                ch, 
                authority=user_authority_pk, 
                subaccount_id=x, 
                use_cache=True
            )
            try:
                await chu_sub.set_cache()
            except:
                continue
            summary = {}
            summary['total_collateral'] = (await chu_sub.get_total_collateral())/1e6
            summary['initial_margin_req'] = await chu_sub.get_margin_requirement(MarginCategory.INITIAL)
            summary['maintenance_margin_req'] = await chu_sub.get_margin_requirement(MarginCategory.MAINTENANCE)
            all_summarys.append(pd.DataFrame(summary, index=[x]))

        sub_dfs = pd.concat(all_summarys)
        sub_dfs.index.name = 'subaccount_id'
        st.dataframe(sub_dfs)

    else:
        st.text('not found')

    
    