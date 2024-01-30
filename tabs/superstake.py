
import sys
import driftpy
import pandas as pd 
import numpy as np 
import plotly.express as px

pd.options.plotting.backend = "plotly"
from driftpy.accounts.oracle import get_oracle_price_data_and_slot
import plotly.graph_objects as go

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.math.perp_position import calculate_position_funding_pnl
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import *
import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import matplotlib.pyplot as plt 
from driftpy.drift_user import get_token_amount
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
    
    utilization =  borrows/(deposits+1e-9) * 100
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

@st.cache_data(ttl=3600*2)  # 2 hr TTL this time
def get_stake_yield():
    metrics = requests.get('https://api2.marinade.finance/metrics_json').json()
    apy = metrics['msol_price_apy_30d']
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

def apy_to_apr(apy, compounds_per_year):
    compound_amount = 1 + apy/100
    est_apr = (compound_amount ** (1/float(compounds_per_year)) - 1) * compounds_per_year
    return est_apr

async def super_stake(clearing_house: DriftClient):
    c1, c2, c3 = st.columns(3)

    ch = clearing_house
    # state = await get_state_account(ch.program)
    msol_market = await get_spot_market_account(ch.program, 2)
    sol_market = await get_spot_market_account(ch.program, 1)

    msol_oracle = (await clearing_house.get_oracle_price_data_and_slot(ch.program.provider.connection, msol_market.oracle, msol_market.oracle_source)).data
    sol_oracle = (await get_oracle_price_data_and_slot(ch.program.provider.connection, sol_market.oracle, sol_market.oracle_source)).data


    msol_price, stake_apr = get_stake_yield()
    stake_apr = apy_to_apr(stake_apr, int(365/2))

    collat = c1.number_input('mSOL:', min_value=1e-9, value=1.0, step=1.0)
    
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
    sol_init_lwgt = calc_size_liab(sol_init_lwgt, sol_imf, abs(ss_sol_bor))
    sol_maint_lwgt = calc_size_liab(sol_maint_lwgt, sol_imf, abs(ss_sol_bor))

    msol_dep, msol_bor = get_ir_curves(msol_market, ss_msol_dep)


    sol_dep, sol_bor = get_ir_curves(sol_market, -ss_sol_bor)
    c3.write(f'`mSOL` deposit: `{ss_msol_dep:,.9f}`')
    c3.write(f'`SOL` borrow: `{ss_sol_bor:,.9f}`')
    divdown = msol_price * ((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep*msol_price))
    c3.write(f'Liq Price:`{divdown:,.4f}` (current = `{msol_price:,.4f}` `mSOL/SOL`)')
    # st.write(stake_apr, msol_dep)
    tabs = st.tabs(['reward', 'risk'])

    with tabs[0]:
        s1, s2 = st.columns(2)
        s1.metric('normal stake apr:', f'{float(stake_apr*100):,.4f}%')
        # st.write((ss_msol_dep * (stake_apr+msol_dep))*msol_price, ss_sol_bor*sol_bor)
        aa = (ss_msol_dep * (stake_apr+msol_dep))*msol_price - (ss_sol_bor*sol_bor)
        aa /= collat*msol_price
        # st.write(aa)


        if aa-stake_apr > 0:
            s2.metric('super stake apr:', 
                    f'{float(aa *100):,.4f}%',
                        f'{float((aa-stake_apr) * 100):,.4f}% more profitable'
                    )
        else:
            s2.metric('super stake apr:', 
                        f'{float(aa * 100):,.4f}%',
                            f'{float((aa-stake_apr) * 100):,.4f}% less profitable'
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

        byeq = st.radio('by:', ['leverage', 'size'], horizontal=True)
        ans = None
        if byeq == 'leverage':
            for x_lev in levs:
                ss_msol_dep = (collat * x_lev)
                ss_sol_bor = (ss_msol_dep-collat) * msol_price

                msol_init_awgt =  msol_market.initial_asset_weight/1e4
                msol_maint_awgt = sol_market.maintenance_asset_weight/1e4
                sol_init_lwgt = sol_market.initial_liability_weight/1e4
                sol_maint_lwgt = sol_market.maintenance_liability_weight/1e4


                msol_init_awgt = calc_size_ast(msol_init_awgt, msol_imf, ss_msol_dep)
                msol_maint_awgt = calc_size_ast(msol_maint_awgt, msol_imf, ss_msol_dep)
                sol_init_lwgt = calc_size_liab(sol_init_lwgt, sol_imf, abs(ss_sol_bor))
                sol_maint_lwgt = calc_size_liab(sol_maint_lwgt, sol_imf, abs(ss_sol_bor))

                divup = abs((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep) - msol_price)*100

                divup = (msol_maint_awgt*ss_msol_dep*msol_price/(sol_maint_lwgt*ss_sol_bor) - 1) *100
                divdown = -((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep*msol_price) - 1)*100
                dd1.append(divdown)
                dd1[0] = np.nan
                dd2.append(divup)
            ans = pd.DataFrame([dd1, dd2], index=['mSOL depegs down', 'SOL depegs up'], columns=levs ).T
        elif byeq == 'size':
            sizes = list(np.linspace(1, 100000, 100))
            for x_size in sizes:
                ss_msol_dep = (x_size * lev)
                ss_sol_bor = (ss_msol_dep-x_size) * msol_price

                msol_init_awgt =  msol_market.initial_asset_weight/1e4
                msol_maint_awgt = sol_market.maintenance_asset_weight/1e4
                sol_init_lwgt = sol_market.initial_liability_weight/1e4
                sol_maint_lwgt = sol_market.maintenance_liability_weight/1e4


                msol_init_awgt = calc_size_ast(msol_init_awgt, msol_imf, ss_msol_dep)
                msol_maint_awgt = calc_size_ast(msol_maint_awgt, msol_imf, ss_msol_dep)
                sol_init_lwgt = calc_size_liab(sol_init_lwgt, sol_imf, abs(ss_sol_bor))
                sol_maint_lwgt = calc_size_liab(sol_maint_lwgt, sol_imf, abs(ss_sol_bor))


                divup = abs((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep) - msol_price)*100

                divup = (msol_maint_awgt*ss_msol_dep*msol_price/(sol_maint_lwgt*ss_sol_bor) - 1) *100
                divdown = -((sol_maint_lwgt*ss_sol_bor)/(msol_maint_awgt*ss_msol_dep*msol_price) - 1)*100
                dd1.append(divdown)
                # dd1[0] = np.nan
                dd2.append(divup)
            ans = pd.DataFrame([dd1, dd2], index=['mSOL depegs down', 'SOL depegs up'], columns=sizes ).T

        st.write('"depegs" means while one asset (mSOL or SOL) stays in same in value, the other one diverges')
        st.write('note that mSOL depegging higher and SOL depegging lower do not risk this position')
        # fig = ans.plot(log_y=True)
        # fig.update_layout(
        #     title="Super Stake Liquidation Risk",
        #     xaxis_title=byeq,
        #     yaxis_title="% change in price",
        
        #     )
        # aaa = ans.loc[float(lev)-1e-6:].values[0]
        # if byeq == 'leverage':
        #     fig.add_vline(x=lev, line_color="green", annotation_text='leverage \n'+str(aaa))
        
        # st.plotly_chart(fig)

        z_data = pd.DataFrame(index=ans.index)
        z_data['SOL_liq_price'] = (sol_oracle.price + ans['SOL depegs up']*sol_oracle.price/100)/1e6
        z_data['mSOL_liq_price'] = (msol_oracle.price - ans['mSOL depegs down']*msol_oracle.price/100)/1e6
        # fig2 = z_data.plot(log_y=True)
        # aaa = z_data.loc[float(lev)-1e-6:].values[0]
        # fig2.add_vline(x=lev, line_color="green", annotation_text='liq price \n'+str(aaa))
        # # fig2.add_hline(y=(float(msol_oracle.price)/1e6), line_color="blue", annotation_text='mSOL price')
        # # fig.add_hline(y=(sol_oracle.price/1e6), line_color="blue", annotation_text='SOL price')
        
        # st.write('Note: these liquidation prices assume other asset is unchanged thus is a "depeg"')
        # fig2.update_layout(
        #     title="Super Stake Liquidation Price",
        #     xaxis_title=byeq,
        #     yaxis_title="Liquidation Price",
        
        #     )


        msolratioliq = (msol_oracle.price/1e6)/z_data['SOL_liq_price']
        msolratioliq.values[0] = np.nan
        msolratioliq.name = 'mSOL/SOL liq price'
        fig3 = msolratioliq.plot()
        curpx = float(msol_oracle.price)/sol_oracle.price
        fig3.add_hline(y=curpx, line_color="green", annotation_text='current mSOL/SOL price = '+str(curpx)[:8])
        
        xtit = byeq
        if byeq == 'size':
            xtit = f'mSOL size (@ {lev}x leverage)'
        elif byeq == 'leverage':
            xtit = f'leverage (with {collat} mSOL)'
        fig3.update_layout(
            title="Super Stake Liquidation Price",
            xaxis_title=xtit,
            yaxis_title="Liquidation Price",
    
        )
        st.plotly_chart(fig3)


        # fig2 = go.Figure(data=[go.Surface(z=z_data.values)])
        # fig2.update_layout(scene=dict(zaxis=dict(dtick=1, type='log')))

        # fig2.add_vline(x=lev, line_color="green", annotation_text='leverage \n'+str(aaa))
        # # fig2.update_layout(scene=dict(yaxis=dict(dtick=1, type='log')))

        # st.plotly_chart(fig2)

    # with tabs[2]:
    #     doit = st.radio('load "Super Stake SOL" users:', [True, False], horizontal=True)

    #     if doit:
    #         from solana.rpc.types import MemcmpOpts
    #         from driftpy.drift_user import DriftUser
    #         from driftpy.types import UserAccount
    #         from driftpy.math.margin import MarginCategory

    #         all_users = ch.program.account['User'].all(filters=[MemcmpOpts(offset=4267, bytes='2i')])
    #         if all_users is not None:
    #             fuser: User = all_users[0].account
    #             chu = DriftUser(
    #                 ch, 
    #                 authority=fuser.authority, 
    #                 sub_account_id=fuser.sub_account_id, 
    #                 use_cache=True
    #             )
    #             await chu.set_cache()
    #             cache = chu.CACHE
    #             for x in all_users:
    #                 key = str(x.public_key)
    #                 account: User = x.account

    #                 chu = DriftUser(ch, authority=account.authority, sub_account_id=account.sub_account_id, use_cache=True)
    #                 cache['user'] = account # update cache to look at the correct user account
    #                 await chu.set_cache(cache)
    #                 margin_category = MarginCategory.INITIAL
    #                 total_liability = await chu.get_margin_requirement(margin_category, None)
    #                 spot_value = await chu.get_spot_market_asset_value(None, False, None)
    #                 upnl = await chu.get_unrealized_pnl(False, None, None)
    #                 total_asset_value = await chu.get_total_collateral(None)
    #                 leverage = await chu.get_leverage()