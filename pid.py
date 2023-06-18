
import sys
import driftpy
import pandas as pd 
import numpy as np 
import plotly.express as px

pd.options.plotting.backend = "plotly"

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
from driftpy.math.margin import MarginCategory
from driftpy.types import SpotBalanceType
import plotly.express as px
from solana.rpc.types import MemcmpOpts

async def show_pid_positions(clearing_house: ClearingHouse):
    ch = clearing_house
    state = await get_state_account(ch.program)

    with st.expander('state'):
        st.json(state.__dict__)
    
    col1, col2, col3, col4 = st.columns(4)

    see_user_breakdown = col1.radio('see users breakdown:', ['All', 'Active', None], 2)

    all_users = None

    if see_user_breakdown is not None:
        try:
            if see_user_breakdown == 'All':
                all_users = await ch.program.account['User'].all()
            else:
                all_users = await ch.program.account['User'].all(memcmp_opts=[MemcmpOpts(offset=4350, bytes='1')])
            
        except Exception as e:
            print('ERRRRR:', e)
            st.write("ERROR: cannot load ['User'].all() with current rpc")
            return
    

    authorities = set()
    dfs = {}
    spotdfs = {}
    # healths = []
    from driftpy.types import User
    kp = Keypair()
    ch = ClearingHouse(ch.program, kp)

    
    perp_liq_prices = {}
    spot_liq_prices = {}

    perp_liq_deltas = {}

    if all_users is not None:
        fuser: User = all_users[0].account
        chu = ClearingHouseUser(
            ch, 
            authority=fuser.authority, 
            subaccount_id=fuser.sub_account_id, 
            use_cache=True
        )
        await chu.set_cache()
        cache = chu.CACHE

        for x in all_users:
            key = str(x.public_key)
            account: User = x.account

            chu = ClearingHouseUser(ch, authority=account.authority, subaccount_id=account.sub_account_id, use_cache=True)
            cache['user'] = account # update cache to look at the correct user account
            await chu.set_cache(cache)
            margin_category = MarginCategory.INITIAL
            total_liability = await chu.get_margin_requirement(margin_category, None)
            spot_value = await chu.get_spot_market_asset_value(None, False, None)
            upnl = await chu.get_unrealized_pnl(False, None, None)
            total_asset_value = await chu.get_total_collateral(None)
            leverage = await chu.get_leverage()

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
                    perp_market = await chu.get_perp_market(pos.market_index)
                    # try:
                    oracle_price = (await chu.get_perp_oracle_data(perp_market)).price / PRICE_PRECISION
                    # except:
                    #     oracle_price = perp_market.amm.historical_oracle_data.last_oracle_price / PRICE_PRECISION
                    liq_price = await chu.get_perp_liq_price(pos.market_index)
                    upnl = await chu.get_unrealized_pnl(False, pos.market_index)
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

                if pos.scaled_balance != 0:
                    spot_market = await chu.get_spot_market(pos.market_index)
                    try:
                        oracle_price = (await chu.get_spot_oracle_data(spot_market)).price / PRICE_PRECISION
                    except:
                        oracle_price = spot_market.historical_oracle_data.last_oracle_price / PRICE_PRECISION
                    liq_price = await chu.get_spot_liq_price(pos.market_index)
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
        
    markettype = col2.radio("MarketType", ('Perp', 'Spot'))
    if markettype == 'Perp':
        num_markets = state.number_of_markets
    else:
        num_markets = state.number_of_spot_markets

    tabs = st.tabs([str(x) for x in range(num_markets)])
    usdc_market = await get_spot_market_account(ch.program, 0)

    for market_index, tab in enumerate(tabs):
        market_index = int(market_index)
        with tab:
            if markettype == 'Perp':
                market = await get_perp_market_account(ch.program, market_index)
                market_name = ''.join(map(chr, market.name)).strip(" ")

                with st.expander(markettype+" market market_index="+str(market_index)+' '+market_name):
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
                    perp_market = await chu.get_perp_market(market_index)
                    try:
                        oracle_price = (await chu.get_perp_oracle_data(perp_market)).price/PRICE_PRECISION
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

            else:
                market = await get_spot_market_account(ch.program, market_index)
                market_name = ''.join(map(chr, market.name)).strip(" ")

                with st.expander(markettype+" market market_index="+str(market_index)+' '+market_name):
                    mdf = serialize_spot_market(market).T
                    st.table(mdf)

                conn = ch.program.provider.connection
                ivault_pk = market.insurance_fund.vault
                svault_pk = market.vault
                iv_amount = int((await conn.get_token_account_balance(ivault_pk))['result']['value']['amount'])
                sv_amount = int((await conn.get_token_account_balance(svault_pk))['result']['value']['amount'])

                token_scale = (10**market.decimals)
                if market_index == 0:
                    col3.metric(f'{market_name} vault balance', str(sv_amount/token_scale) , str(iv_amount/token_scale)+' (insurance)')
                else:
                    col4.metric(f'{market_name} vault balance', str(sv_amount/token_scale) , str(iv_amount/token_scale)+' (insurance)')
                
                PERCENTAGE_PRECISION = 10**6
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
                        'cumulative_deposits',
                        'open_orders',
                        'liq_price_delta',
                        'authority',
                    ]
                    if market_index == 0:
                        columns.pop(columns.index('liq_price_delta'))

                    st.dataframe(df1[columns])

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
                utilization =  borrows/deposits

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
                    spot_market = await chu.get_spot_market(market_index)
                    try:
                        oracle_price = await chu.get_spot_oracle_data(spot_market).price/PRICE_PRECISION
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
        