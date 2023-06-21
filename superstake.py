
import sys
import driftpy
import pandas as pd 
import numpy as np 
import plotly.express as px

pd.options.plotting.backend = "plotly"
from driftpy.clearing_house_user import get_oracle_data
import plotly.graph_objects as go

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.math.positions import calculate_position_funding_pnl
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
import matplotlib.pyplot as plt 
from driftpy.clearing_house_user import get_token_amount
from driftpy.types import SpotBalanceType
import plotly.express as px
import requests
PERCENTAGE_PRECISION = 1e6


def get_ir_curves(market, delt):
    deposits =market.deposit_balance * market.cumulative_deposit_interest/1e10/(1e9)
    borrows = market.borrow_balance * market.cumulative_borrow_interest/1e10/(1e9)
    
    # st.warning(f'{deposits} {borrows} {delt}')
    if delt < 0:
        borrows += abs(delt)
    else:
        deposits += delt
    
    utilization =  borrows/deposits * 100
    market_name = bytes(market.name).decode('utf-8')

    if utilization > 100:
        st.error(f'{market_name} exceeds max borrows')
        utilization = 100

    opt_util = market.optimal_utilization/PERCENTAGE_PRECISION * 100
    opt_borrow = market.optimal_borrow_rate/PERCENTAGE_PRECISION
    max_borrow = market.max_borrow_rate/PERCENTAGE_PRECISION

    bor_rate =  opt_borrow* (100/opt_util)*utilization/100 if utilization <= opt_util else ((max_borrow-opt_borrow) * (100/opt_util))*(utilization-opt_util)/100 + opt_borrow
    
    dep_rate = bor_rate*utilization*(1-market.insurance_fund.total_factor/1e6)/100
    return dep_rate, bor_rate

@st.experimental_memo(ttl=3600*2)  # 2 hr TTL this time
def get_stake_yield():
    metrics = requests.get('https://api2.marinade.finance/metrics_json').json()
    apy = metrics['avg_staking_apy']
    msol_price = metrics['m_sol_price']
    return msol_price, apy

def calc_size_ast(wgt, imf, base):
    if imf == 0:
        return wgt
    
    dd = np.sqrt(np.abs(base))
    res = min(wgt, 1.1 / (1 + (imf * dd)))
    return res


def calc_size_liab(wgt, imf, base):
    liability_wgt_n = wgt
    if(imf != 0):
        liability_wgt_n = wgt * .8 #/(1/(imf/10000))
    dd = np.sqrt(np.abs(base))
    res = max(wgt, liability_wgt_n + imf * dd)
    return res

async def super_stake(clearing_house: ClearingHouse):
    ch = clearing_house
    # state = await get_state_account(ch.program)
    msol_market = await get_spot_market_account(ch.program, 2)
    sol_market = await get_spot_market_account(ch.program, 1)

    msol_oracle = await get_oracle_data(ch.program.provider.connection, msol_market.oracle, msol_market.oracle_source)
    sol_oracle = await get_oracle_data(ch.program.provider.connection, sol_market.oracle, sol_market.oracle_source)


    msol_price, stake_apy = get_stake_yield()
    stake_apy = stake_apy/100

    c1, c2 = st.columns(2)
    collat = c1.number_input('msol:', min_value=1.0, value=1.0, step=1.0)
    
    msol_imf = msol_market.imf_factor/1e6
    sol_imf = sol_market.imf_factor/1e6



    msol_init_awgt =  msol_market.initial_asset_weight/1e4
    msol_maint_awgt = sol_market.maintenance_asset_weight/1e4
    sol_init_lwgt = sol_market.initial_liability_weight/1e4
    sol_maint_lwgt = sol_market.maintenance_liability_weight/1e4



    max_lev = msol_init_awgt/(sol_init_lwgt-1) - 1
    liq_lev = msol_maint_awgt/(sol_maint_lwgt-1) - 1

    lev = c2.slider('leverage', 1.0, float(max_lev), step=.05)
    ss_msol_dep = (collat * lev)
    ss_sol_bor = (ss_msol_dep-collat) * msol_price
    s1, s2 = st.columns(2)

    msol_init_awgt = calc_size_ast(msol_init_awgt, msol_imf, ss_msol_dep)
    msol_maint_awgt = calc_size_ast(msol_maint_awgt, msol_imf, ss_msol_dep)
    sol_init_lwgt = calc_size_liab(sol_init_lwgt - 1, sol_imf, abs(ss_sol_bor)) + 1
    sol_maint_lwgt = calc_size_liab(sol_maint_lwgt - 1, sol_imf, abs(ss_sol_bor)) + 1

    msol_dep, msol_bor = get_ir_curves(msol_market, ss_msol_dep)


    sol_dep, sol_bor = get_ir_curves(sol_market, -ss_sol_bor)
    st.write(f'`mSOL` deposit: `{ss_msol_dep:,.9f}`')
    st.write(f'`SOL` borrow: `{ss_sol_bor:,.9f}`')

    # st.write(stake_apy, msol_dep)
    tabs = st.tabs(['reward', 'risk'])

    with tabs[0]:
        s1, s2 = st.columns(2)
        s1.metric('normal stake apy:', f'{float(stake_apy*100):,.4f}%')
        # st.write((ss_msol_dep * (stake_apy+msol_dep))*msol_price, ss_sol_bor*sol_bor)
        aa = (ss_msol_dep * (stake_apy+msol_dep))*msol_price - (ss_sol_bor*sol_bor)
        aa /= collat*msol_price
        # st.write(aa)

        if aa-stake_apy > 0:
            s2.metric('super stake apy:', 
                    f'{float(aa *100):,.4f}%',
                        f'{float((aa-stake_apy) * 100):,.4f}% more profitable'
                    )
        else:
            s2.metric('super stake apy:', 
                        f'{float(aa * 100):,.4f}%',
                            f'{float((aa-stake_apy) * 100):,.4f}% less profitable'
                        ) 

        entercol, exitcol = st.columns(2)
        entercol.write('''
        ### how to enter
        1) deposit `mSOL`
        2) borrow `SOL`
        3) stake `SOL` for `mSOL`
        4) re-deposit `mSOL`
        ''')

        exitcol.write('''
        ### how to exit
        1) deposit `SOL` (requires excess capital)
        2) withdraw `mSOL`
        3) unstake `mSOL` for `SOL`

        or

        1) swap `mSOL` for `SOL` (spread + fees)
        ''')

        with st.expander('details:'):
            st.write('`mSOL` initial asset weight:', msol_init_awgt, '`SOL` initial liability weight', sol_init_lwgt, )
            st.write('`mSOL` maint asset weight:', msol_maint_awgt, '`SOL` maint liability weight', sol_maint_lwgt, )

            st.write('`SOL`/`mSOL` oracle prices:', sol_oracle.price/1e6, msol_oracle.price/1e6)
            st.write('`mSOL` fair value:', sol_oracle.price/1e6*msol_price)
            st.write('initial weighted', msol_init_awgt*ss_msol_dep*msol_price - sol_init_lwgt*ss_sol_bor)
            st.write('maint weighted', msol_maint_awgt*ss_msol_dep*msol_price - sol_maint_lwgt*ss_sol_bor)
            st.write('msol deposit/borrow apr:', msol_dep, msol_bor)
            st.write('sol deposit/borrow apr:', sol_dep, sol_bor)
            if ss_sol_bor > 0:
                divup = (msol_maint_awgt*ss_msol_dep*msol_price/(sol_maint_lwgt*ss_sol_bor) - 1) *100
                divdown = abs((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep*msol_price) - 1)*100
                st.write('liq leverage:', liq_lev)
                st.warning(f'''liquidation when `mSOL` price diverges down by: `{divdown:,.4f}%`  
                        
                        ({msol_oracle.price/1e6:,.4f} → {msol_oracle.price*(1-divdown/100)/1e6:,.4f})
                        ''')
                st.warning(f'''liquidation when `SOL` price diverges up by: `{divup:,.4f}%`
                        
                        ({sol_oracle.price/1e6:,.4f} → {sol_oracle.price*(1+divup/100)/1e6:,.4f})
                        '''
                        )
                
    with tabs[1]:
        dd1 = []
        dd2 = []
        levs = list(np.linspace(1, 3, 210))
        for x_lev in levs:
            ss_msol_dep = (collat * x_lev)
            ss_sol_bor = (ss_msol_dep-collat) * msol_price
            divup = abs((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep) - msol_price)*100

            divup = (msol_maint_awgt*ss_msol_dep*msol_price/(sol_maint_lwgt*ss_sol_bor) - 1) *100
            divdown = -((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep*msol_price) - 1)*100
            dd1.append(divdown)
            dd1[0] = np.nan
            dd2.append(divup)

        st.write('"depegs" means while one asset (mSOL or SOL) stays in same in value, the other one diverges')
        st.write('note that mSOL depegging higher and SOL depegging lower do not risk this position')
        ans = pd.DataFrame([dd1, dd2], index=['mSOL depegs down', 'SOL depegs up'], columns=levs ).T
        fig = ans.plot(log_y=True)
        fig.update_layout(
            title="Super Stake Liquidation Risk",
            xaxis_title="Leverage",
            yaxis_title="% change in price",
        
            )
        aaa = ans.loc[float(lev)-1e-6:].values[0]
        fig.add_vline(x=lev, line_color="green", annotation_text='leverage \n'+str(aaa))
        
        st.plotly_chart(fig)

        z_data = pd.DataFrame(index=ans.index)
        z_data['SOL_liq_price'] = (sol_oracle.price + ans['SOL depegs up']*sol_oracle.price/100)/1e6
        z_data['mSOL_liq_price'] = (msol_oracle.price - ans['mSOL depegs down']*msol_oracle.price/100)/1e6
        fig2 = z_data.plot(log_y=True)
        aaa = z_data.loc[float(lev)-1e-6:].values[0]
        fig2.add_vline(x=lev, line_color="green", annotation_text='liq price \n'+str(aaa))
        # fig2.add_hline(y=(float(msol_oracle.price)/1e6), line_color="blue", annotation_text='mSOL price')
        # fig.add_hline(y=(sol_oracle.price/1e6), line_color="blue", annotation_text='SOL price')
        
        st.write('Note: these liquidation prices assume other asset is unchanged thus is a "depeg"')
        fig2.update_layout(
            title="Super Stake Liquidation Price",
            xaxis_title="Leverage",
            yaxis_title="Liquidation Price",
        
            )
        st.plotly_chart(fig2)

        # fig2 = go.Figure(data=[go.Surface(z=z_data.values)])
        # fig2.update_layout(scene=dict(zaxis=dict(dtick=1, type='log')))

        # fig2.add_vline(x=lev, line_color="green", annotation_text='leverage \n'+str(aaa))
        # # fig2.update_layout(scene=dict(yaxis=dict(dtick=1, type='log')))

        # st.plotly_chart(fig2)


