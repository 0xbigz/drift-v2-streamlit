
import sys
import driftpy
import pandas as pd 
import numpy as np 
import plotly.express as px

pd.options.plotting.backend = "plotly"
from driftpy.accounts import DataAndSlot

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser as DriftUser
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
from driftpy.addresses import get_user_account_public_key

import asyncio
import matplotlib.pyplot as plt 
from driftpy.drift_user import get_token_amount
from driftpy.math.margin import MarginCategory
from driftpy.types import SpotBalanceType
import plotly.express as px
from solana.rpc.types import MemcmpOpts
from datafetch.transaction_fetch import load_token_balance
MARGIN_PRECISION = 1e4
PERCENTAGE_PRECISION = 10**6

from driftpy.drift_client import AccountSubscriptionConfig

async def show_pid_positions(clearing_house: DriftClient):
    ch = clearing_house
    await ch.account_subscriber.update_cache()
    state = ch.get_state_account()

    with st.expander('state'):
        st.json(state.__dict__)
    
    col1, col2, col3, col4 = st.columns(4)

    see_user_breakdown = col1.radio('see users breakdown:', ['All', 'Active', 'OpenOrder', 'SuperStakeSOL', 'SuperStakeJitoSOL', None], 5)

    all_users = None

    if see_user_breakdown is not None:
        try:
            if see_user_breakdown == 'All':
                all_users = await ch.program.account['User'].all()
            elif see_user_breakdown == 'Active':
                all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=4350, bytes='1')])
            elif see_user_breakdown == 'OpenOrder':
                all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=4352, bytes='2')])          
            elif see_user_breakdown == 'SuperStakeSOL':
                all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=72, bytes='3LRfP5UkK8aDLdDMsJS3D')])
            elif see_user_breakdown == 'SuperStakeJitoSOL':
                all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=72, bytes='GHB8xrCziYmaX9fbpnLFAMBVby')])
        except Exception as e:
            print('ERRRRR:', e)
            st.write("ERROR: cannot load ['User'].all() with current rpc")
            return
    

    authorities = set()
    dfs = {}
    spotdfs = {}
    # healths = []
    from driftpy.types import UserAccount
    kp = Keypair()
    # ch = DriftClient(ch.program, kp)

    
    perp_liq_prices = {}
    spot_liq_prices = {}

    perp_liq_deltas = {}

    if all_users is not None:
        fuser: DriftUser = all_users[0].account
        user_account_pk = get_user_account_public_key(
                    clearing_house.program_id,
                    fuser.authority,
                    fuser.sub_account_id)
        chu = DriftUser(
            ch, 
            user_account_pk
            # use_cache=True
        )
        await chu.drift_client.account_subscriber.update_cache()
        cache = chu.drift_client.account_subscriber.cache
        for x in all_users:
            key = str(x.public_key)
            account: DriftUser = x.account

            user_account_pk = get_user_account_public_key(
                    clearing_house.program_id,
                    account.authority,
                    account.sub_account_id)

            chu = DriftUser(ch, user_account_pk,
                            account_subscription=AccountSubscriptionConfig("cached"))
                            # use_cache=True
                            
            chu.account_subscriber.user_and_slot = DataAndSlot(0, account)
            chu.drift_client.account_subscriber.cache = cache

            # chu = DriftUser(ch, authority=account.authority, sub_account_id=account.sub_account_id, 
            #                 # use_cache=True
            #                 )
            # cache['user'] = account # update cache to look at the correct user account
            # await chu.set_cache(cache)
            margin_category = MarginCategory.INITIAL
            total_liability = chu.get_margin_requirement(margin_category, None)
            spot_value = chu.get_spot_market_asset_value(None, False, None)
            upnl = chu.get_unrealized_pnl(False, None, None)
            total_asset_value = chu.get_total_collateral(None)
            leverage = chu.get_leverage()

            total_position_size = 0
            pos: PerpPosition
            for idx, pos in enumerate(x.account.perp_positions):
                total_position_size += abs(pos.base_asset_amount)

            # mr = await chu.get_margin_requirement('Maintenance')
            # tc = await chu.get_total_collateral('Maintenance')
            # health = (tc-mr)
            # healths.append(health)

            if account.authority not in authorities:
                authorities.add(account.authority)

            dfs[key] = []
            name = str(''.join(map(chr, account.name)))

            from driftpy.types import PerpPosition, SpotPosition
            pos: PerpPosition
            for idx, pos in enumerate(x.account.perp_positions):
                dd = pos.__dict__
                dd['position_index'] = idx
                dd['authority'] = str(x.account.authority)
                dd['name'] = name 
                dd['spot_value'] = spot_value/QUOTE_PRECISION
                dd['total_liability'] = total_liability / QUOTE_PRECISION
                dd['total_asset_value'] = total_asset_value / QUOTE_PRECISION
                dd['leverage'] = leverage / MARGIN_PRECISION
                if pos.base_asset_amount != 0:
                    perp_market = chu.get_perp_market_account(pos.market_index)
                    try:
                        oracle_price = (chu.get_oracle_data_for_perp_market(perp_market)).price / PRICE_PRECISION
                    except:
                        oracle_price = perp_market.amm.historical_oracle_data.last_oracle_price / PRICE_PRECISION
                    liq_price = chu.get_perp_liq_price(pos.market_index)
                    upnl = chu.get_unrealized_pnl(False, pos.market_index)
                    upnl_funding = calculate_position_funding_pnl(perp_market, pos)

                    if liq_price is None: 
                        liq_delta = None
                    else:
                        perp_liq_prices[pos.market_index] = perp_liq_prices.get(pos.market_index, []) + [liq_price]
                        liq_delta = liq_price - oracle_price
                        perp_liq_deltas[pos.market_index] = perp_liq_deltas.get(pos.market_index, []) + [(liq_delta, abs(pos.base_asset_amount), leverage/10_000)]

                else: 
                    liq_delta = None
                    upnl = 0
                    upnl_funding = 0

                dd['liq_price_delta'] = liq_delta
                dd['upnl'] = upnl
                dd['funding_upnl'] = upnl_funding

                dfs[key].append(pd.Series(dd))
            dfs[key] = pd.concat(dfs[key],axis=1) 
            
            spotdfs[key] = []
            pos: SpotPosition
            for idx, pos in enumerate(x.account.spot_positions):
                dd = pos.__dict__
                dd['position_index'] = idx
                dd['authority'] = str(x.account.authority)
                dd['name'] = name
                dd['spot_value'] = spot_value/QUOTE_PRECISION
                dd['total_liability'] = total_liability / QUOTE_PRECISION
                dd['total_asset_value'] = total_asset_value / QUOTE_PRECISION
                dd['leverage'] = leverage / MARGIN_PRECISION

                if pos.scaled_balance != 0:
                    spot_market = ch.get_spot_market_account(pos.market_index)
                    try:
                        oracle_price = (chu.get_oracle_data_for_spot_market(spot_market)).price / PRICE_PRECISION
                    except:
                        oracle_price = spot_market.historical_oracle_data.last_oracle_price / PRICE_PRECISION
                    liq_price = chu.get_spot_liq_price(pos.market_index)
                    if liq_price is None: 
                        liq_delta = None
                    else:
                        spot_liq_prices[pos.market_index] = spot_liq_prices.get(pos.market_index, []) + [liq_price]
                        liq_delta = liq_price - oracle_price
                else: 
                    liq_delta = None

                dd['liq_price_delta'] = liq_delta

                spotdfs[key].append(pd.Series(dd))
            spotdfs[key] = pd.concat(spotdfs[key],axis=1)    

        user_leaderboard = pd.DataFrame([x.account for x in all_users])
        user_leaderboard = user_leaderboard[['authority', 'sub_account_id', 'settled_perp_pnl',  'cumulative_perp_funding', 'total_deposits',
        'total_withdraws', 'total_social_loss',
        'cumulative_spot_fees',
        'next_order_id',
        'next_liquidation_id', 
        'status',
        'is_margin_trading_enabled',]].sort_values('settled_perp_pnl', ascending=False).reset_index(drop=True)

        for x in ['total_deposits',
        'total_withdraws', 'total_social_loss',  'settled_perp_pnl',
        'cumulative_spot_fees', 'cumulative_perp_funding',]:
            user_leaderboard[x]/=1e6

        st.text('user leaderboard')
        st.dataframe(user_leaderboard)

    col3.metric("Unique Driftoors", str((state.number_of_authorities)),str((state.number_of_sub_accounts))+" SubAccounts")

    if all_users is not None:
        perps = pd.concat(dfs,axis=1).T
        perps.index = perps.index.set_names(['public_key', 'idx2'])
        perps = perps.reset_index()

        spots = pd.concat(spotdfs,axis=1).T
        spots.index = spots.index.set_names(['public_key', 'idx2'])
        spots = spots.reset_index()
        
    num_perp_markets = state.number_of_markets
    num_spot_markets = state.number_of_spot_markets

    cat_tabs = st.tabs(['overview', 'perp (%i)' % num_perp_markets, 'spot (%i)' % num_spot_markets])
    usdc_market = ch.get_spot_market_account(0)
    usdc_dep_rate = 0

    with cat_tabs[0]:
        po_deposits = 0
        po_position_notional = 0
        po_position_notional_net = 0
        est_funding_dol = 0
        dd1 = {}
        dd2 = {}
        for market_index in range(state.number_of_markets):
            perp_i = ch.get_perp_market_account(market_index)
            market_name = ''.join(map(chr, perp_i.name)).strip(" ")

            fee_pool = (perp_i.amm.fee_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
            pnl_pool = (perp_i.pnl_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
            po_deposits += fee_pool + pnl_pool

            otwap = perp_i.amm.historical_oracle_data.last_oracle_price_twap
            market_price_spread = (perp_i.amm.last_mark_price_twap - otwap)
            funding_offset = otwap / 5000 # ~ 7% annual premium
            pred_fund = ((market_price_spread + funding_offset)/otwap) / (3600.0/perp_i.amm.funding_period * 24)
            
            fundings = [x* 100 * 365.25 * 24 for x in (pred_fund,
                        perp_i.amm.last_funding_rate / otwap / FUNDING_RATE_BUFFER,
                        perp_i.amm.last24h_avg_funding_rate / otwap / FUNDING_RATE_BUFFER,
                        )]

            dd1[market_name] = fundings

            po_position_notional_i = (perp_i.amm.base_asset_amount_with_amm+perp_i.amm.base_asset_amount_with_unsettled_lp)/1e9 * otwap/1e6
            po_position_notional += abs(po_position_notional_i)
            po_position_notional_net += (po_position_notional_i)
            est_funding_dol += pred_fund * po_position_notional_i


        total_borrow_fee_annualized = 0
        for market_index in range(state.number_of_spot_markets):
            market = ch.get_spot_market_account(market_index)
            market_name = ''.join(map(chr, market.name)).strip(" ")
            deposits = market.deposit_balance * market.cumulative_deposit_interest/1e10/(1e9)
            borrows = market.borrow_balance * market.cumulative_borrow_interest/1e10/(1e9)

            utilization =  borrows/(deposits+1e-12) * 100
            opt_util = market.optimal_utilization/PERCENTAGE_PRECISION * 100
            opt_borrow = market.optimal_borrow_rate/PERCENTAGE_PRECISION
            max_borrow = market.max_borrow_rate/PERCENTAGE_PRECISION

            bor_ir_curve = [
                opt_borrow* (100/opt_util)*x/100 
                if x <= opt_util
                else ((max_borrow-opt_borrow) * (100/(100-opt_util)))*(x-opt_util)/100 + opt_borrow
                for x in [utilization]
            ]

            borrow_fee_annualized = borrows * bor_ir_curve[0] * market.insurance_fund.total_factor / 1e6
            borrow_fee_notional_annualized = borrow_fee_annualized * market.historical_oracle_data.last_oracle_price/1e6

            total_borrow_fee_annualized += borrow_fee_notional_annualized


            dep_ir_curve = [ir*utilization*(1-market.insurance_fund.total_factor/1e6)/100 for idx,ir in enumerate(bor_ir_curve)]
            if market_index == 0:
                usdc_dep_rate = dep_ir_curve[0]
            dd2[market_name] = (dep_ir_curve[0]*100, bor_ir_curve[0]*100, utilization, borrow_fee_notional_annualized)
        s1, s2 = st.columns(2)
        s1.write(pd.DataFrame(dd1, index=['predicted funding rate', 'last funding rate', '24h avg funding rate']).T)
        s2.write(pd.DataFrame(dd2, index=['deposit rate', 'borrow rate', 'utilization', 'annualized_borrow_revenue']).T)
        
        p1, p2 = st.columns(2)

        ann_factor = 365 * 24

        p1.metric('protocol-owned position', f'${po_position_notional:,.2f}', f'${est_funding_dol*ann_factor:,.2f} interest per year')
        p1.metric('protocol-owned net position', f'${po_position_notional_net:,.2f}', f'${abs(po_position_notional_net*.05):,.2f} at risk on 5% move')
        
        p2.metric('protocol-owned deposits', f'${po_deposits:,.2f}', f'${usdc_dep_rate*po_deposits:,.2f} interest per year')
        p2.metric('est borrow revenue', f'${total_borrow_fee_annualized:,.2f}', f'${total_borrow_fee_annualized/365:,.2f} interest per day')

    with cat_tabs[1]:
        tabs = st.tabs([str(x) for x in range(num_perp_markets)])
        for market_index, tab in enumerate(tabs):
            market_index = int(market_index)
            with tab:
                market = ch.get_perp_market_account(market_index)
                market_name = ''.join(map(chr, market.name)).strip(" ")

                with st.expander('Perp'+" market market_index="+str(market_index)+' '+market_name):
                    mdf = serialize_perp_market_2(market).T
                    st.dataframe(mdf)
                
                df1 = None
                if all_users is not None:
                    df1 = perps[((perps.base_asset_amount!=0) 
                                | (perps.quote_asset_amount!=0)
                                |  (perps.lp_shares!=0)
                                |  (perps.open_orders!=0)
                                ) 
                                & (perps.market_index==market_index)
                            ].sort_values('base_asset_amount', ascending=False).reset_index(drop=True)

                pct_long = market.amm.base_asset_amount_long / (market.amm.base_asset_amount_long - market.amm.base_asset_amount_short + 1e-10) 
                
                st.text('User Perp Positions  (Base /|Quote) '+ str((market.number_of_users_with_base)) +' / ' + str((market.number_of_users)))

                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.text(f'user long: {np.round(pct_long*100, 2) }%')
                my_bar = cc1.progress(pct_long)
                imbalance = (market.amm.base_asset_amount_with_amm) /1e9
                cc1.text(f'user imbalance: {np.round(imbalance, 2) } base')
                my_bar2 = cc1.progress(abs(market.amm.base_asset_amount_with_amm)/((market.amm.base_asset_amount_long - market.amm.base_asset_amount_short + 1e-10)))

                # st.text('user long % sentiment:')
                # sentiment = 0
                # if len(df1):
                #     sentiment = df1['base_asset_amount'].pipe(np.sign).sum()/len(df1) + .5
                # my_bar = st.progress(sentiment)
                st.text(f'vAMM Liquidity (bids= {(market.amm.max_base_asset_reserve-market.amm.base_asset_reserve) / 1e9} | asks={(market.amm.base_asset_reserve-market.amm.min_base_asset_reserve) / 1e9})')
                t0, t1, t2 = st.columns([1,1,5])
                dir = t0.selectbox('direction:', ['buy', 'sell'], key='selectbox-'+str(market_index))
                ba = t1.number_input('base amount:', value=1, key='numin-'+str(market_index))
                bid_price = market.amm.bid_quote_asset_reserve/market.amm.bid_base_asset_reserve * market.amm.peg_multiplier/1e6
                ask_price = market.amm.ask_quote_asset_reserve/market.amm.ask_base_asset_reserve * market.amm.peg_multiplier/1e6
                def px_impact(dir, ba):
                    f = ba / (market.amm.base_asset_reserve/1e9)
                    if dir == 'buy':
                        pct_impact = (1/((1-f)**2) - 1) * 100
                    else:
                        pct_impact = (1 - 1/((1+f)**2)) * 100
                    return pct_impact
                px_impact = px_impact(dir, ba)
                price = (ask_price * (1+px_impact/100)) if dir=='buy' else (bid_price * (1-px_impact/100))
                t2.text(f'vAMM stats: \n px={price} \n px_impact={px_impact}%')
            

                rev_pool = usdc_market.revenue_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10/(1e9)

                fee_pool = (market.amm.fee_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
                pnl_pool = (market.pnl_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
                excess_pnl = fee_pool+pnl_pool - market.amm.quote_asset_amount/1e6 + (market.amm.base_asset_amount_with_amm + market.amm.base_asset_amount_with_unsettled_lp)/1e9  * market.amm.historical_oracle_data.last_oracle_price/1e6
                
                dfff = pd.DataFrame({'pnl_pool': pnl_pool, 'fee_pool': fee_pool, 'rev_pool': rev_pool}, index=[0]).T.reset_index()
                dfff['color'] = 'balance'

                df_flow = pd.DataFrame({
                'pnl_pool': market.amm.net_revenue_since_last_funding/1e6,
                'fee_pool': market.insurance_claim.revenue_withdraw_since_last_settle/1e6, 
                'rev_pool': -(market.insurance_claim.revenue_withdraw_since_last_settle/1e6)
                }, index=[0]
                ).T.reset_index()
                df_flow['color'] = 'flows'

                df_flow_max = pd.DataFrame({
                'pnl_pool': market.unrealized_pnl_max_imbalance/1e6,
                'fee_pool': market.insurance_claim.max_revenue_withdraw_per_period/1e6, 
                'rev_pool': -(market.insurance_claim.max_revenue_withdraw_per_period/1e6)
                }, index=[0]
                ).T.reset_index()
                df_flow_max['color'] = 'max_flow'
                dfff = pd.concat([dfff, df_flow, df_flow_max]).reset_index(drop=True)
                # print(dfff)
                fig = px.funnel(dfff, y='index', x=0, color='color')
                st.plotly_chart(fig)

                st.text(f'Revenue Withdrawn Since Settle Ts: {market.insurance_claim.revenue_withdraw_since_last_settle/1e6}')
                st.text(f'''
                Last Settle Ts: {str(pd.to_datetime(market.insurance_claim.last_revenue_withdraw_ts*1e9))} vs
                Last Spot Settle Ts: {str(pd.to_datetime(usdc_market.insurance_fund.last_revenue_settle_ts*1e9))}
                
                ''')

                st.text(f'Ext. Insurance: {(market.insurance_claim.quote_max_insurance-market.insurance_claim.quote_settled_insurance)/1e6} ({market.insurance_claim.quote_settled_insurance/1e6}/{market.insurance_claim.quote_max_insurance/1e6})')
                st.text(f'Int. Insurance: {fee_pool}')
                st.text(f'PnL Pool: {pnl_pool}')
                st.text(f'Excess PnL: {excess_pnl} ({fee_pool+pnl_pool} - {market.amm.quote_asset_amount/1e6 + (market.amm.base_asset_amount_with_amm/1e9 * market.amm.historical_oracle_data.last_oracle_price/1e6)})')

                if df1 is not None:
                    df1['base_asset_amount'] /= 1e9
                    df1['remainder_base_asset_amount'] /= 1e9
                    df1['lp_shares'] /= 1e9
                    df1['quote_asset_amount'] /= 1e6
                    df1['quote_entry_amount'] /= 1e6
                    df1['quote_break_even_amount'] /= 1e6
                    df1['upnl'] /= 1e6
                    df1['funding_upnl'] /= 1e6

                    df1['entry_price'] = -df1['quote_entry_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                    df1['breakeven_price'] = -df1['quote_break_even_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                    df1['cost_basis'] = -df1['quote_asset_amount']/df1['base_asset_amount'].apply(lambda x: -1 if x==0 else x)


                    agg_desc = df1.sum()

                    toshow = df1[[
                        'authority', 
                        'name', 
                        'open_orders', 
                        'total_liability',
                        'spot_value',
                        'total_asset_value',
                        'leverage',
                        'base_asset_amount', 
                        'liq_price_delta',
                        'lp_shares',
                        'remainder_base_asset_amount',
                        'entry_price', 
                        'breakeven_price', 
                        'cost_basis',
                        'upnl',
                        'funding_upnl',
                        'public_key', 
                    ]]
                    st.dataframe(toshow)
                    with st.expander('user summary stats'):
                        st.dataframe(agg_desc)


                # visualize perp liquidations 
                perp_liq_prices_m = perp_liq_prices.get(market_index, None)
                if perp_liq_prices_m is not None and len(perp_liq_prices_m) > 0:
                    perp_market = chu.get_perp_market_account(market_index)
                    try:
                        oracle_price = (chu.get_perp_oracle_data(perp_market)).price/PRICE_PRECISION
                    except:
                        oracle_price = perp_market.amm.historical_oracle_data.last_oracle_price / PRICE_PRECISION

                    st.markdown('## Liquidation Prices/Sizes')
                    max_price = int(max(np.median(perp_liq_prices_m), oracle_price) * 1.3)
                    max_price = st.number_input('max_price', value=max_price, key=1337*(market_index+1))

                    _perp_liq_prices_m = []
                    for p in perp_liq_prices_m: 
                        if p < max_price: 
                            _perp_liq_prices_m.append(p)
                    perp_liq_prices_m = _perp_liq_prices_m
                    n_bins = st.number_input('number of bins:', value=min(100, len(perp_liq_prices_m)), key=1*(market_index+1))

                    df = pd.DataFrame({'liq_price': perp_liq_prices_m})
                    fig = px.histogram(perp_liq_prices_m, nbins=n_bins, labels={'x': 'liq_price', 'y':'count'})
                    fig.add_vline(x=oracle_price, line_color="red", annotation_text='oracle price')
                    fig = fig.update_layout( 
                        xaxis_title="Liquidation Price",
                        yaxis_title="# of Users",
                    )
                    st.plotly_chart(fig)

                    liq_deltas, sizes, leverages = zip(*perp_liq_deltas[market_index])
                    pos, neg = [], []
                    for l in liq_deltas:
                        if l > 0: pos.append(l)
                        if l < 0: neg.append(l)
                    max_pos_liq = np.median(pos)
                    max_neg_liq = np.median(neg)

                    st.write('larger bubble = larger position size')
                    st.text(f'liquidation delta range: {max_neg_liq} - {max_pos_liq}')
                    _liq_deltas = []
                    for liq in liq_deltas: 
                        if liq > 0: _liq_deltas.append(min(max_pos_liq, liq))
                        if liq < 0: _liq_deltas.append(max(max_neg_liq, liq))
                    liq_deltas = _liq_deltas

                    df = pd.DataFrame({
                        'liq_delta': liq_deltas, 
                        'position_size': [s/AMM_RESERVE_PRECISION for s in sizes],
                        'leverage': leverages,
                    })
                    fig = px.scatter(df, x='liq_delta', y='leverage', size='position_size', color='leverage')
                    fig.add_vline(x=0, line_color="red", annotation_text='liquidation')

                    st.plotly_chart(fig)

                else: 
                    st.write("no liquidations found...")


    with cat_tabs[2]:
        tabs = st.tabs([str(x) for x in range(num_spot_markets)])
        for market_index, tab in enumerate(tabs):
            market_index = int(market_index)
            with tab:
                market = ch.get_spot_market_account(market_index)
                market_name = ''.join(map(chr, market.name)).strip(" ")
                spot_col1, = st.columns(1)
                with st.expander('Spot'+" market market_index="+str(market_index)+' '+market_name):
                    mdf = serialize_spot_market(market).T
                    st.table(mdf)

                conn = ch.program.provider.connection
                ivault_pk = market.insurance_fund.vault
                svault_pk = market.vault
                iv_amount = await load_token_balance(conn, ivault_pk)
                sv_amount = await load_token_balance(conn, svault_pk)

                token_scale = (10**market.decimals)
                spot_col1.metric(f'{market_name} vault balance', str(sv_amount/token_scale) , str(iv_amount/token_scale)+' (insurance)')
   
                df1 = None
                if all_users is not None:
                    df1 = spots[(spots.scaled_balance!=0) & (spots.market_index==market_index)
                            ].sort_values('scaled_balance', ascending=False).reset_index(drop=True)\
                                .drop(['idx2', 'padding'], axis=1)
                    for col in ['scaled_balance']:
                        df1[col] /= 1e9
                    for col in ['cumulative_deposits']:
                        df1[col] /= (10 ** market.decimals)

                    st.text('User Spot Positions ('+ str(len(df1)) +')')
                    columns = [
                        'public_key',
                        'balance_type',
                        'scaled_balance',
                        'spot_value',
                        'leverage',
                        'cumulative_deposits',
                        'open_orders',
                        'liq_price_delta',
                        'authority',
                    ]
                    if market_index == 0:
                        columns.pop(columns.index('liq_price_delta'))

                    st.dataframe(df1[columns])

                    df111 = pd.concat({
                                'leverage': (df1['leverage']).astype(float)*4.5,
                                'position_size': df1['spot_value'].astype(float),
                                'position_size2': (df1['spot_value'].astype(float)+1).pipe(np.log),
                            },axis=1)
                    try:
                        fig111 = px.scatter(df111, x='leverage', y='position_size', size='position_size2', size_max=20, color='position_size', log_y=True)
                        st.plotly_chart(fig111)
                    except Exception as e:
                        st.write('cannot do scatter plot... (%s)' % str(e))

                    total_cumm_deposits = df1['cumulative_deposits'].sum()

                    sb = df1[df1['balance_type'].map(str) == 'SpotBalanceType.Deposit()']
                    deposits = sb['scaled_balance'].values 
                    total_deposits = 0
                    for depo in deposits: 
                        token_amount = get_token_amount(
                            depo * 1e9, 
                            market, 
                            'SpotBalanceType.Deposit()'
                        )
                        total_deposits += token_amount
                            
                    sb = df1[df1['balance_type'].map(str) == 'SpotBalanceType.Borrow()']
                    borrows = sb['scaled_balance'].values 
                    total_borrows = 0
                    for borr in borrows: 
                        token_amount = get_token_amount(
                            borr * 1e9, 
                            market, 
                            'SpotBalanceType.Borrow()'
                        )
                        total_borrows += token_amount

                    total_deposits /= (10 ** market.decimals)
                    total_borrows /= (10 ** market.decimals)
                    
                    st.text(f'total deposits (token amount): {total_deposits:,.2f}')
                    st.text(f'total borrows (token amount): {total_borrows:,.2f}')
                    st.text(f'total cummulative deposits: {total_cumm_deposits:,.2f}')


                    fig1, ax1 = plt.subplots()
                    fig1.set_size_inches(15.5, 5.5)
                    if total_deposits + total_borrows > 0:
                        ax1.pie([total_deposits, total_borrows], labels=['deposits', 'borrows'], autopct='%1.5f%%',
                                startangle=90)
                        ax1.axis('equal')  
                        st.pyplot(fig1)

                opt_util = market.optimal_utilization/PERCENTAGE_PRECISION * 100
                opt_borrow = market.optimal_borrow_rate/PERCENTAGE_PRECISION
                max_borrow = market.max_borrow_rate/PERCENTAGE_PRECISION

                ir_curve_index = [x/100 for x in range(0,100*100+1, 10)]
                bor_ir_curve = [
                    opt_borrow* (100/opt_util)*x/100 
                    if x <= opt_util
                    else ((max_borrow-opt_borrow) * (100/(100-opt_util)))*(x-opt_util)/100 + opt_borrow
                    for x in ir_curve_index
                ]

                dep_ir_curve = [ir*ir_curve_index[idx]*(1-market.insurance_fund.total_factor/1e6)/100 for idx,ir in enumerate(bor_ir_curve)]

                ir_fig = (pd.DataFrame([dep_ir_curve, bor_ir_curve], 
                index=['deposit interest', 'borrow interest'], 
                columns=ir_curve_index).T * 100).plot() 

                deposits = market.deposit_balance * market.cumulative_deposit_interest/1e10/(1e9)
                borrows = market.borrow_balance * market.cumulative_borrow_interest/1e10/(1e9)
                utilization =  borrows/(deposits+1e-12)

                ir_fig.add_vline(x=market.utilization_twap/1e6 * 100, line_color="blue", annotation_text='util_twap')
                ir_fig.add_vline(x=utilization * 100 , line_color="green", annotation_text='util')

                ir_fig.update_layout(
                    title=market_name+" Interest Rate",
                    xaxis_title="utilization (%)",
                    yaxis_title="interest rate (%)",
                    legend_title="Curves",
                )
                st.plotly_chart(ir_fig)

                # visualize spot liquidations 
                st.write('## Liquidation Prices')
                spot_liq_prices_m = spot_liq_prices.get(market_index, None)
                if spot_liq_prices_m is not None and len(spot_liq_prices_m) > 0 and market_index != 0: # usdc (assumed to always be 1) doesnt really make sense
                    spot_market = chu.get_spot_market_account(market_index)
                    try:
                        oracle_price = chu.get_spot_oracle_data(spot_market).price/PRICE_PRECISION
                    except:
                        oracle_price = spot_market.historical_oracle_data.last_oracle_price / PRICE_PRECISION
                        
                    st.markdown('## Liquidation Prices')
                    max_price = st.number_input('max_price', value=max(oracle_price, np.median(spot_liq_prices_m)) * 1.3, key=19*(market_index+1))

                    _spot_liq_prices_m = []
                    for p in spot_liq_prices_m: 
                        if p < max_price: 
                            _spot_liq_prices_m.append(p)
                    spot_liq_prices_m = _spot_liq_prices_m

                    n_bins = st.number_input('number of bins:', value=min(100, len(spot_liq_prices_m)), key=23*(market_index+1))
                    df = pd.DataFrame({'liq_price': spot_liq_prices_m})
                    fig = px.histogram(spot_liq_prices_m, nbins=n_bins, labels={'x': 'liq_price', 'y':'count'})
                    fig = fig.update_layout( 
                        xaxis_title="Liquidation Price",
                        yaxis_title="# of Users",
                    )
                    fig.add_vline(x=oracle_price, line_color="red", annotation_text='oracle price')

                    st.plotly_chart(fig)
                else: 
                    st.write("no liquidations found...")



    if df1 is not None:
        authority = st.text_input('public_key:') 
        auth_df = df1[df1['public_key'] == authority]
        if len(auth_df) != 0: 
            columns = list(auth_df.columns)
            if markettype == 'Perp':
                columns.pop(columns.index('idx2'))
            df = auth_df[columns].iloc[0]
            json_df = df.to_json()
            st.json(json_df)
        else: 
            st.markdown('public key not found...')
        