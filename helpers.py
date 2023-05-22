
from driftpy.math.amm import *
from driftpy.math.trade import *
from driftpy.math.positions import *
from driftpy.math.market import *
from driftpy.math.user import *

from driftpy.types import *
from driftpy.constants.numeric_constants import *

from driftpy.setup.helpers import mock_oracle, _airdrop_user, set_price_feed, set_price_feed_detailed, adjust_oracle_pretrade, _mint_usdc_tx
from driftpy.admin import Admin
from driftpy.types import OracleSource

from driftpy.clearing_house import ClearingHouse as SDKClearingHouse
from driftpy.math.amm import calculate_mark_price_amm
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account

from anchorpy import Provider, Program, create_workspace, WorkspaceType
import pprint
import os
import json
import pandas as pd

from solana.keypair import Keypair

from subprocess import Popen
import datetime
import subprocess
from solana.transaction import Transaction
import asyncio
from tqdm import tqdm


def human_amm_df(df):
    bool_fields = [ 'last_oracle_valid']
    enum_fields = ['oracle_source']
    pure_fields = ['last_update_slot', 'long_intensity_count', 'short_intensity_count', 
    'curve_update_intensity', 'amm_jit_intensity'
    ]
    reserve_fields = [
        'base_asset_reserve', 'quote_asset_reserve', 'min_base_asset_reserve', 'max_base_asset_reserve', 'sqrt_k',
        'ask_base_asset_reserve', 'ask_quote_asset_reserve', 'bid_base_asset_reserve', 'bid_quote_asset_reserve',
        'terminal_quote_asset_reserve', 'base_asset_amount_long', 'base_asset_amount_short', 'base_asset_amount_with_amm', 'base_asset_amount_with_unsettled_lp',
        'user_lp_shares', 'min_order_size', 'max_position_size', 'order_step_size', 'max_open_interest',
        ]

    wgt_fields = ['initial_asset_weight', 'maintenance_asset_weight',
    
    'initial_liability_weight', 'maintenance_liability_weight',
    'unrealized_pnl_initial_asset_weight', 'unrealized_pnl_maintenance_asset_weight']

    pct_fields = ['base_spread','long_spread', 'short_spread', 'max_spread', 'concentration_coef',
    'last_oracle_reserve_price_spread_pct',
    'last_oracle_conf_pct',
        #spot market ones
    'utilization_twap',

    'imf_factor', 'unrealized_pnl_imf_factor', 'liquidator_fee', 'if_liquidation_fee',
    'optimal_utilization', 'optimal_borrow_rate', 'max_borrow_rate',
    ]

    funding_fields = ['cumulative_funding_rate_long', 'cumulative_funding_rate_short', 'last_funding_rate', 'last_funding_rate_long', 'last_funding_rate_short', 'last24h_avg_funding_rate']
    quote_asset_fields = ['total_fee', 'total_mm_fee', 'total_exchange_fee', 'total_fee_minus_distributions',
    'total_fee_withdrawn', 'total_liquidation_fee', 'cumulative_social_loss', 'net_revenue_since_last_funding',
    'quote_asset_amount_long', 'quote_asset_amount_short', 'quote_entry_amount_long', 'quote_entry_amount_short',
    'volume24h', 'long_intensity_volume', 'short_intensity_volume',
    'total_spot_fee', 'quote_asset_amount',
    'quote_break_even_amount_short', 'quote_break_even_amount_long'
    ]
    time_fields = ['last_trade_ts', 'last_mark_price_twap_ts', 'last_oracle_price_twap_ts', 'last_index_price_twap_ts',]
    duration_fields = ['lp_cooldown_time', 'funding_period']
    px_fields = [
        'last_oracle_normalised_price',
        'order_tick_size',
        'last_bid_price_twap', 'last_ask_price_twap', 'last_mark_price_twap', 'last_mark_price_twap5min',
    'peg_multiplier',
    'mark_std',
    'oracle_std',
    'last_oracle_price_twap', 'last_oracle_price_twap5min',
    'last_oracle_price', 'last_oracle_conf', 

    #spot market ones
        'last_index_bid_price', 'last_index_ask_price', 'last_index_price_twap', 'last_index_price_twap5min',
    
    ]
    token_fields = ['deposit_token_twap', 'borrow_token_twap', 'max_token_deposits', 'withdraw_guard_threshold']
    balance_fields = ['scaled_balance', 'deposit_balance', 'borrow_balance']
    interest_fileds = ['cumulative_deposit_interest', 'cumulative_borrow_interest']
    for col in df.columns:
        # if col in enum_fields or col in bool_fields:
        #     pass
        # else if col in duration_fields:
        #     pass
        # else if col in pure_fields:
        #     pass
        if col in reserve_fields:
            df[col] /= 1e9
        elif col in funding_fields:
            df[col] /= 1e9
        elif col in wgt_fields:
            df[col] /= 1e4
        elif col in quote_asset_fields:
            df[col] /= 1e6
        elif col in pct_fields:
            df[col] /= 1e6
        elif col in px_fields:
            df[col] /= 1e6
        elif col in token_fields:
            z = df['decimals'].values[0]
            df[col] /= (10**z)
        elif col in interest_fileds:
            df[col] /= 1e10
        elif col in time_fields:
            df[col] = [datetime.datetime.fromtimestamp(x) for x in df[col].values]
        elif col in balance_fields:
            df[col] /= 1e9
            
    return df

def human_market_df(df):
    enum_fields = ['status', 'contract_tier', '']
    pure_fields = ['number_of_users', 'market_index', 'next_curve_record_id', 'next_fill_record_id', 'next_funding_rate_record_id']
    pct_fields = ['imf_factor', 'unrealized_pnl_imf_factor', 'liquidator_fee', 'if_liquidation_fee']
    wgt_fields = ['initial_asset_weight', 'maintenance_asset_weight',
    
    'initial_liability_weight', 'maintenance_liability_weight',
    'unrealized_pnl_initial_asset_weight', 'unrealized_pnl_maintenance_asset_weight']
    margin_fields = ['margin_ratio_initial', 'margin_ratio_maintenance']
    px_fields = [
        'expiry_price',
        'last_oracle_normalised_price',
        'order_tick_size',
        'last_bid_price_twap', 'last_ask_price_twap', 'last_mark_price_twap', 'last_mark_price_twap5min',
    'peg_multiplier',
    'mark_std',
    'oracle_std',
    'last_oracle_price_twap', 'last_oracle_price_twap5min',
    
    ]
    time_fields = ['last_trade_ts', 'expiry_ts', 'last_revenue_withdraw_ts']
    balance_fields = ['scaled_balance', 'deposit_balance', 'borrow_balance']
    quote_fields = [
        'total_spot_fee', 
        'unrealized_pnl_max_imbalance', 'quote_settled_insurance', 'quote_max_insurance', 
    'max_revenue_withdraw_per_period', 'revenue_withdraw_since_last_settle', ]
    token_fields = ['borrow_token_twap', 'deposit_token_twap', 'withdraw_guard_threshold', 'max_token_deposits']
    interest_fields = ['cumulative_deposit_interest', 'cumulative_borrow_interest']

    for col in df.columns:
        # if col in enum_fields:
        #     pass
        # elif col in pure_fields:
        #     pass
        if col in pct_fields:
            df[col] /= 1e6
        elif col in px_fields:
            df[col] /= 1e6
        elif col in margin_fields:
            df[col] /= 1e4
        elif col in wgt_fields:
            df[col] /= 1e4
        # elif col in time_fields:
        #     pass
        elif col in quote_fields:
            df[col] /= 1e6
        elif col in balance_fields:
            df[col] /= 1e9
        elif col in interest_fields:
            df[col] /= 1e10
        elif col in token_fields:
            df[col] /= 1e6 #todo   

    return df
    

def serialize_perp_market_2(market: PerpMarket):

    market_df = pd.json_normalize(market.__dict__).drop(['amm', 'insurance_claim', 'pnl_pool'],axis=1).pipe(human_market_df)
    market_df.columns = ['market.'+col for col in market_df.columns]

    amm_df= pd.json_normalize(market.amm.__dict__).drop(['historical_oracle_data', 'fee_pool'],axis=1).pipe(human_amm_df)
    amm_df.columns = ['market.amm.'+col for col in amm_df.columns]

    amm_hist_oracle_df= pd.json_normalize(market.amm.historical_oracle_data.__dict__).pipe(human_amm_df)
    amm_hist_oracle_df.columns = ['market.amm.historical_oracle_data.'+col for col in amm_hist_oracle_df.columns]

    market_amm_pool_df = pd.json_normalize(market.amm.fee_pool.__dict__).pipe(human_amm_df)
    market_amm_pool_df.columns = ['market.amm.fee_pool.'+col for col in market_amm_pool_df.columns]

    market_if_df = pd.json_normalize(market.insurance_claim.__dict__).pipe(human_market_df)
    market_if_df.columns = ['market.insurance_claim.'+col for col in market_if_df.columns]

    market_pool_df = pd.json_normalize(market.pnl_pool.__dict__).pipe(human_amm_df)
    market_pool_df.columns = ['market.pnl_pool.'+col for col in market_pool_df.columns]

    result_df = pd.concat([market_df, amm_df, amm_hist_oracle_df, market_amm_pool_df, market_if_df, market_pool_df],axis=1)
    return result_df

def serialize_spot_market(spot_market: SpotMarket):
    spot_market_df = pd.json_normalize(spot_market.__dict__).drop([
        'historical_oracle_data', 'historical_index_data',
        'insurance_fund', # todo
        'spot_fee_pool', 'revenue_pool'
        ], axis=1).pipe(human_amm_df)
    spot_market_df.columns = ['spot_market.'+col for col in spot_market_df.columns]

    if_df= pd.json_normalize(spot_market.insurance_fund.__dict__).pipe(human_amm_df)
    if_df.columns = ['spot_market.insurance_fund.'+col for col in if_df.columns]

    hist_oracle_df= pd.json_normalize(spot_market.historical_oracle_data.__dict__).pipe(human_amm_df)
    hist_oracle_df.columns = ['spot_market.historical_oracle_data.'+col for col in hist_oracle_df.columns]

    hist_index_df= pd.json_normalize(spot_market.historical_index_data.__dict__).pipe(human_amm_df)
    hist_index_df.columns = ['spot_market.historical_index_data.'+col for col in hist_index_df.columns]


    market_pool_df = pd.json_normalize(spot_market.revenue_pool.__dict__).pipe(human_amm_df)
    market_pool_df.columns = ['spot_market.revenue_pool.'+col for col in market_pool_df.columns]


    market_fee_df = pd.json_normalize(spot_market.spot_fee_pool.__dict__).pipe(human_amm_df)
    market_fee_df.columns = ['spot_market.spot_fee_pool.'+col for col in market_fee_df.columns]

    result_df = pd.concat([spot_market_df, if_df, hist_oracle_df, hist_index_df, market_pool_df, market_fee_df],axis=1)
    return result_df
