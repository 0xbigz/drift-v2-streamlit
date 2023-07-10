

import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.math.oracle import *
import datetime

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType, SpotPosition, PerpPosition
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
import plotly.graph_objs as go
async def userdataraw(clearing_house: ClearingHouse):
    # connection = clearing_house.program.provider.connection

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            # st.write(type(obj))
            # st.write(str(type(obj)))
            if 'Position' in str(type(obj)) or 'Order' in str(type(obj)):
                return obj.__dict__
            elif isinstance(obj, PublicKey):
                return str(obj)
            else:
                return str(obj)
            return super().default(obj)
    
    inp = st.text_input('user account:', )
    if len(inp)>5:
        st.write(inp)
        st.write(PublicKey(str(inp)))
        user = (await clearing_house.program.account["User"].fetch(PublicKey(str(inp))))
        # st.write(user.__dict__['spot_positions'])
        st.json(json.dumps(user.__dict__, cls=CustomEncoder))
        # st.json(user)




