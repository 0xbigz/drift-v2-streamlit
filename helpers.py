
from driftpy.math.amm import *
from driftpy.math.trade import *
from driftpy.math.perp_position import *
from driftpy.math.market import *
from driftpy.math.user import *

from driftpy.types import *
from driftpy.constants.numeric_constants import *

from driftpy.setup.helpers import mock_oracle, _airdrop_user, set_price_feed, set_price_feed_detailed, adjust_oracle_pretrade, _mint_usdc_tx
from driftpy.admin import Admin
from driftpy.types import OracleSource

from driftpy.drift_client import DriftClient, AccountSubscriptionConfig
from driftpy.math.amm import calculate_mark_price_amm
from driftpy.drift_user import DriftUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account

from anchorpy import Provider, Program, create_workspace, WorkspaceType
import pprint
import os
import json
import pandas as pd

from solders.keypair import Keypair

from subprocess import Popen
import datetime
import subprocess
from solana.transaction import Transaction
import asyncio
from tqdm import tqdm
from driftpy.math.margin import MarginCategory
import copy
from driftpy.accounts import DataAndSlot


# over ~100k in value
DRIFT_WHALE_LIST_SNAP = '''BRksHqLiq2gvQw1XxsZq6DXZjD3GB5a9J63tUBgd6QS9
7tDm5mxdUcqW423mX9a3yC9MkPQccaTYuPmAxZBkGNxn
ETsaTf7tsaYEChnLfu2iwzFXWxJbCzwD6bPkMGCyB42d
9e3dJLadqDVdunTLbQwW5rPyq2284pXJ6aWQtKD6DZHJ
7SeykJkVT24ZkuNwyYNACWj5JaLdNRjxVeBD5LAJbMmB
2aMcirYcF9W8aTFem6qe8QtvfQ22SLY6KUe6yUQbqfHk
4oTeSjNig62yD4KCehU4jkpNVYowLfaTie6LTtGbmefX
8RLb4ys91TjY3EqWHEVEbNnzzSbQGsdwn6Dj3dtY8Z67
3KdSdkCjSPvpthQScANfNRkSfX8xpqcngTTUnnZJoxUm
8LsnV5g8xtKcjiwyBBUNZonyaiC91GcbX2sXazVefVyX
2oJLLjUnoq6dvwdQKBWAoY2cUJKCEgMBs49HxJiRKwDY
2Crr5X53x36xcBUpEV52wT1Xbnhknwaz2XVjeTXTJEUY
BsZ7DBAqNiTC9PN5St4cb8aUoaCzGB23NRq6Xcv2CfUW
DQmcci1JpEPHkXYoAn3kKkDB7ZNpEQjZPUbQd97z1aXM
73ZwKGCsaMitEcTQV4W5tBnyht9XayUmShA9xbzG8EUG
DiDsuvdxGUQdvLBToYN4Bmk2GfFSR2bFFMRu7GhmMkJG
HmrXYdAiraCKSjyuTuzX24Jb6QDNj36oTD1BwWb369Gd
EHQ2YfkMQ9UTV3yCLTuMXqWNfoLSuwBnrwNZ2iUPGHcF
HpFLzDwaKqfHWZGGHgUJfVRB3D2RF4EsY9NKaFXcTxRr
GPmMKkE3cuvoDmQmyPdqqoxhWXeZZAcbJ726r1zbBEkE
8us4LosdD7Ct2jQEHFsfZdF3Uaj9mtwYbBwpxeWvcaGp
EU7uEtXojSPz1FtmaJnUrLJkx9RTwayS3TyfveBZ5qDC
HbwQHQ52NZZTuSgpWdSrm3zHKFB4VKSrufWVH5bV7b8C
BsVi8TcSsfdZuoksw1enwrDTYEVRnPBze4ZSPSqmafqh
F1vmkcTwgC7AVSQkVMpRhjws1Ni3gmz4Cu55Xe1TWXvA
G4iis9dLtpKwFkERDUBUqjbWPY4uJwYkGnWcZQUa1mpv
3hH27WiZV2QmM6bK5Rz7bkrqHK2PX7wNo4EogPzmGadN
56BnDPDBv8yhJdsVjkuH53QjvjiMA4qURKuDy7FYWX2X
AM1zueGrHg7zQDYUfVWtzFgn2G6EAuX161KMrkSgtkkh
22n9s2gQQN5T5t1i8RrHpKju5RkiJsw2LrfmgBsdPhmn
2EUQ6kxy2cM558ZJN4D31x6RGgLwgK1eo9BAEFjochVT
FgbsjcwKyUYLKLoYyq3piHW3xfU7cemPS8GniM1eeoUs
FeCV4X2uSehvYRSUHU7eVqtiJdnFJ2fGShC98CNW8gwG
Daz7CYfAym86pf6MmzsfZFz22wuLNLqECu1wx7rKLxsn
Er5x5oryNKNwvCgPBz2481m82NutLWMHivHBRGJzV9uv
5z3wiSDCPnkxywMj1VJ64ZbvcVPoUK8ydCp74rkeqzZ8
Lshvy96HmpRUyUhx1KP5F1xL2LvJerEpbHSh2xqAmg1
4EUfWFFpxckhBiCRN35wLqQFH6XeT4sRy5NkBF3249Fu
4b7qSeHYDqN3byBgbtsQNVfNcJ7x2P9BLFNk9yQfQqLq
2Mc7TScL3xN4iRRFRpaGJP3zy1Zaq2ZRV21epuobQMHg
EEBjRSo6kNeQXpRYmxuuBcBNB4MqxmRsQaarNUJTHBL5
GC3hdnsPZVkVsXCceF5HyTWxu7ZD7sj7CMiVyPW7ew9y
HvSYehyjXAcL9Dq8DW9sYaUieRWpz5wozB3U5ajCRvqc
4rv4bGgCuKpakw2BTCasnkzSgqi7QFGmWEEkGEw9modm
C13FZykQfLXKuMAMh2iuG7JxhQqd8otujNRAgVETU6id
74obHhSMzZLyyWpbz9SfbrFtKGmmTmAfp13FEV26PDgB
DUfFQC9uXXmSc7amJPr7pwf75LV2oNtSkW5wV669b1LQ
4yN82JynHtd57JB63CepKmCs7nvsiqk4KShjPd9xtpDU
FAmFxKSzAdtDkb8eoHNyGf9Qrk7qHGo6BAZujRe5r1fc
3ZHrooUPNAv7NuajRW4N95t6Pm5Xeyps15q9oeSPt5Q4
3gvf9XBYcs453b1UXDnaj3CgNcz5p9dT9aiYDhxBePiC
7WrK1Ku9qXv2Lpum7osN7n7vrdDPdRorM4LYVDZ2Qsts
8hiFmuXjwWpmUpSJfnZb9jfSCx8gPfqwQUVcNV8WmS8K
46BbA4BYke2wYZsqRXiyXcz67LtSBR911KXoVTuRrDg2
8LNHw4qZGmXAKMJ6qL3PS45yCpJQre49oCzrVVCc76sW
74tj4HwRvWY3yHc7PNRzAuDss4z9udeSygKdfB9HSBtY
EaETWptCSuVjLeMe2c2exYt4CzHLruUZQF3LXTgR87de
G8ipCTW4dn1KwYyMBuTaXWp5stYbtrjYeLA6JxrAgbQe
DwtySoEwL2QSArfmUzVXaVFXW8ZLCHEDuXQCpqpq5V3j
7cBRvpTLiLi6cp3bLaZ8u9AerjrwsqPo37H2jKChnE3c
AMz5hbWe2B2ohM3LhVk5eRRtSYVUgANdUKR3M5Xb4t7s
EcY7jYAZAdgRzM8wBsR2qP3faL25w7C3Uy18HtzWdfZQ
EZe9hRFsUuSFD9kVeAEa8cmKZHsnBvKdGLiPspHBFrB
AQ66FNXphVFfTR8mFmv1PmCgkQzRKUP3Giw4bRYdHERe
6xjDCa6fpehWBQzEdDQW3Gv1kum3VzVKzZpXG4AKckJK
FjnB7KNmjGjYg2vRAQymnePW9wMuaTcP4EiinRcNdZtr
Eh87smZiFo797bjFc5RreRGrmKxvBs2vQM71oNfSxnfA
Gc2wTwfKgr17VzaEaGKHQFGDfqWo3miAshHzNhHRMGPJ
HFk6YDhctF45YcxP4U3JbEMVt4YkmhGz7mNU9nyXMpLe
BdehTirPRjv7iAakdzL3vD41VMYrizaEYYCSA2gVnWv
6X4aNRKmxeh6BLKhzGm8RedaspAigHZJPMAaM6nJdcJu
BoaYJas68C483JJjUaD31M3c1bkuRosFWpZrBfoALnyv
D99njKWrHAJXpkjKxLYLAeYU4z3RuQh6hWGBiCHhxi7i
J7tpzrQHfWhW69Q6GBzW778ProJ3hRLyvaHMmGqsB5ZQ
g1PMk3xmxdb631cFkrCzJuGxehRBqs6P3o7nGAvhnNU
AGFbBosmenBH9fm9SkKcz8m6borRbKKJxUfQnGsHDcsn
HVMMKeksGXZih5aQpv1KYHCBhaGCRuXqE3bkQCURYehU
HnmRSitzQk249TVjTxVmWakNUrdMfEnG6TBpv7V9pQWB
4ywReyHX6hEr3GyP4d1AXpkam8HmFYZeKz8f3xpbqYSN
A5VNoX7JDuzi6BTRkAvEBf2dBiFK67yqbQC7ChUatExj
23h8frAqQwRiRRHF2mE8H6vztEZJ6DTYfJfRxr4d64HV
DjN8mCngZHnhDtKYwFjgQk4cjr6oUanE1GCS9AmBwTcq
5J4wTB2dfBLSJJa5g988N4LM8TNhV71EjjifWZej2oj6
8HACL7BXeJBS1aYYUizQFoxwYBVEiDMxD8QsrNkZwDx4
85QLSQAKhWYcKFQJ8xxfi4gkb82Cx6ito8CXLPCTGLn4
7XX43NrjTHTNPd5LfNmHKKLPhVSjgAV4FKRdXGDUVwgh
7GguEzb2ZgeH1TFukx9VdBKJdpRPsRom56WgFYK6zD9e
A3h5TLA7ij3VnsZJmt2g2piVXDenuMskXLUQ7LgcQZTi
GZSciBrAKKhHMZi6yphPKtvhoTssmKJUaCxxDTaKX8md
4X8f5P9V4Jvzr4sU2H6rTKYjJeDFEX2nxbcB3vmG2q2u
DFUXe5EHxAQLoLiUquuH75rFf5h5kbfK8s2PDrKVs6cK
73taguWJkNYo5mTfA52dxgwoTLN8PP6AFyCVHbSSfycg
B3WuWwm46HAaBT1xRHh8ymoiVrftJ3vuDYVaK5vqoX57
ApdcetQhQqrwvwC6BisX7y1iJJLbhtziPbXwko1KrpTQ
3Q89n76Lx7K6yyVagH4BTRfPJ8BLb5K8wB6LwvDK72WW
AkmPdA5DQsJgLsTiskRyeGbYzGxJ9GxAE6AA2vp9wmtx
9CNuKaJ7zzDvUHQpGmjB4akDYBFv6HXqhmxAQYx9Qw7w
6t4DUWiaewuxfuCo7RpAyw3avcPWp8zvpeogpBfuLFEL
ENZihoaRFsY1ZHzWibR3uMMVhMT2xAcm1JaAgoLxGSte
Bi9U9KvmgQK9kLww3N3UJvPy5A1mJnRqyzm4BXDePHvF
BXP1a8cCBdGNrpTjfncqeSxy8hhX6Szs4P76MF8eDQUU
F4ypfFqdWvR4zFCsbBLv6j29huFjpgRmmCKkVN9M8khT
FmohRubtmmrwWe8xJ3R6WTp5BYyVv6do9V2tfnFijUtg
9xnPxwedXTbwYeDegP4kXQ6LWyNKCwhaJuuF7ST7bxRi
HnH5gaa84M7phQFwNMp2PHVQqdeVCoXKFGu1nyTwKpuB
E3M3VWPsiUp89uxzLw4Cg1hZMbqk5PfHqRQqQ6qj4dX
7WBa9Du2N6KJTNnQdpWG4djRKzkZZcCan6ksEPaX8Vpo
9fwhHftD1KM3C5QQ3TJZBD13MVL4r9s3BsuR7rzbVZcu
Fgfdx9QFvwQwZ4QdUGPUBES7YR5S6itZPVJSUYKh5VEE
HPVVmzTKkwMWbpNyT9PQoeowBEWaXca2g9NYUawYuBQ6
2AkGKesooNq1Hn39xMsZnJkjQaNuqyaXaHfmPHgmpmPL
481L7qoCfkWqutamEcj7AYDS6P3azawF5WhshWRAPVdi
fBcykPptSKYJTCRZ8JTSN53gGo5HyQ9SaJXSpfRMERR
2nBqswD2KW4Vh63ZbFmtdRHT4n649bvS8n2XAueToi3P
59XziNjAka8dh9yDnquThAFu5jdHbhbyanx6h14ZpZTj
DBKiRUXMctVodDxNyQFrr2JJ1rZF6kr4muQDSKhMpBqu
6oSJJuGZSz1UgeJDMMHGMz81NbbXWsBn4P5Jrn3YQJo4'''


async def all_user_stats(all_users, ch: DriftClient, oracle_distort=None, pure_cache=None, only_one_index=None):
    if all_users is not None:
        fuser: DriftUser = all_users[0].account
        chu = DriftUser(
            ch, 
            user_public_key=fuser.user_public_key, 
            # sub_account_id=fuser.sub_account_id, 
            # use_cache=True
        )
        if pure_cache is None and chu.drift_client.account_subscriber.cache is None:
            await chu.drift_client.account_subscriber.update_cache()
        else:
            chu.drift_client.account_subscriber.cache = pure_cache
            
        cache1 = copy.deepcopy(chu.drift_client.account_subscriber.cache)
        if oracle_distort is not None:
            # new_spots = []
            new_oracles_dat = {}

            # for i,x in enumerate(cache1['spot_market_oracles']):
            #     # sol and msol move with
            #     if (i != 0 and only_one_index is None) or (only_one_index == 0 and i in [1,2]):
            #         x.price *= oracle_distort
            #     new_spots.append(x)
            # cache1['spot_market_oracles'] = new_spots

            for i,(key, val) in enumerate(cache1['oracle_price_data'].items()):
                new_oracles_dat[key] = copy.deepcopy(val)
                if only_one_index is None or only_one_index == key:
                    new_oracles_dat[key].data.price *= oracle_distort
                    # if oracle_distort >= 0:
                    #     assert(new_oracles_dat[key].data.price > 0)
            cache1['oracle_price_data'] = new_oracles_dat
            chu.drift_client.account_subscriber.cache = cache1


        res = []
        for x in all_users:
            key = str(x.public_key)
            account: DriftUser = x.account

            chu = DriftUser(ch, user_public_key=account.user_public_key, 
                            # sub_account_id=account.sub_account_id,
                            account_subscription=AccountSubscriptionConfig("cached"))
                            # use_cache=True
                            
            chu.account_subscriber.user_and_slot = DataAndSlot(0, account)
            chu.drift_client.account_subscriber.cache = cache1

            # chu.account_subscriber.
            # cache['user'] = account # update cache to look at the correct user account
            chu.drift_client.account_subscriber.cache = cache1
            margin_category = MarginCategory.INITIAL
            spot_liab = chu.get_spot_market_liability()
            perp_liab = chu.get_perp_market_liability()

            margin_req = chu.get_margin_requirement(margin_category, None)
            spot_value = chu.get_spot_market_asset_value(None, False, None)
            upnl = chu.get_unrealized_pnl(True, only_one_index, None)
            leverage = chu.get_leverage(None)

            res.append([leverage/MARGIN_PRECISION, spot_liab/QUOTE_PRECISION, perp_liab/QUOTE_PRECISION, margin_req/QUOTE_PRECISION, spot_value/QUOTE_PRECISION, upnl/QUOTE_PRECISION])

        res = pd.DataFrame(res, columns=['leverage', 'spot_liability', 'perp_liability', 'margin_requirement', 'spot_value', 'upnl'], index=[x.public_key for x in all_users])
        res['total_liability'] = res['perp_liability']+res['spot_liability']


        return res, chu
    




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
    

def serialize_perp_market_2(market: PerpMarketAccount):

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

def serialize_spot_market(spot_market: SpotMarketAccount):
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
