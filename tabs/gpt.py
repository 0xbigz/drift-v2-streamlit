
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
import csv 
pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
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
from glob import glob
import openai

DRIFT_CONTEXT = """
```
Drift Protocol is an open-sourced, decentralised exchange built on the Solana blockchain, enabling transparent and non-custodial trading on cryptocurrencies.


# Welcome to Drift Protocol


## What is Drift?
Drift Protocol is an open-sourced, decentralised exchange built on the Solana blockchain, enabling transparent and non-custodial trading on cryptocurrencies.
By depositing collateral into Drift Protocol, users can: 
- trade perpetual swaps with up to 10x leverage,
- borrow or lend at variable rate yields, 
- stake / provide liquidity,
- swap spot tokens

## Why use Drift?
The full suite of DeFi tools within the protocol are powered by Drift's robust cross-margined risk engine, designed to give traders a balance of both capital efficiency and protection (more details of the cross-margin engine design are detailed throughout "Technical Explanations").﻿

Under the cross-margin engine, each tool extends functionality within the protocol without over-extending risk. For instance:

the borrow / lend markets also enable cross-collateral on perpetual futures and more efficient margin trading on spot assets
every deposited token is eligible for yield on deposits from borrows and provides margin for perpetual swaps
borrowers are only able eligible to borrow from depositors in an over-collateralised fashion while passing multiple safety measures
The protocol's orderbook, liquidity, and liquidation layer is powered by a validator-like Keeper Network. Keepers are a network of agents and market-makers incentivized to provide the best order execution (i.e. Just-In-Time (JIT) liquidity, order matching, etc.) to traders on Drift. The Keepers can route orders throughout the multi-sourced liquidity mechanisms that are designed to effectively scale and offer competitive pricing even with larger order sizes.

the core contributors are @davidjlu @cindyleow (who do BD & Product), and @bigz_Pubkey @crispheaney (exchange design and smart contracts). @wphan7805, @0xNineteen are backend engineers. @luketheduke__, @StrokeSzn, @evan, @perpetualnudge also front end engineers / product. @chachangnft "damo", @marketducky, @0xMitzy work on business side
"""

INSURANCE_POOL_CONTEXT = """"
    
    ```
    # Drift Accounting: Pools & Settlement

Within Drift Protocol, all token deposits are held in a global collateral vault. This is required for seamless cross-margin and borrow-lend. The only exception to this is the insurance fund vault residing outside.

Ensuring proper accounting across users requires a robust settlement mechanism. The protocol uses intermediate Pool Balances to facilitate transfers and ensure that claimed gains are required to come from settled offsetting losses.


**Perpetual Market**

An individual perpetual market has two pools:

A. P&L Pool: to accumulate funds from users with losses for settlement to users with profits
B. Fee Pool: to accumulate a fraction of exchange fees for the Quote Spot Market's Revenue Pool

The P&L Pool receives the highest priority on claimed funds, in order to give user's the best possible experience. The default fraction of exchange fees for the Fee Pool is total_exchange_fee / 2 and this fraction is determined by: SHARE_OF_FEES_ALLOCATED_TO_CLEARING_HOUSE.

The Fee Pool will only get partially filled up by up to 1% of intermediate P&L settled from a user's losses and aggressively drawn down for the benefit of the P&L Pool otherwise.

**Spot Market**

An individual spot market has two pools:

A. Revenue Pool: to accumulate revenue across the protocol, denominated in this token
      B. Fee Pool: to pay fillers and settle portions to the revenue pool

The Revenue Pool can collect fees from:

- Borrow interest
- Liquidations
- Perpetual Markets

and can pay out to

- Insurance Fund Stakers
- Perpetual Markets

(see details of these rules in Revenue Pool)

The Fee Pool collects exchange fees from swaps and uses them to pay out the Keeper Network Keepers & Decentralised Orderbook

**Future Work**

Currently, a Perpetuals Market can only pull from the Spot Market Revenue Pool and Insurance Fund for its quote currency.

- In the future, it may be possible for a distressed associative perp market (BTC-PERP) to be able to pull funds from the associated spot market (BTC) revenue/insurance pool and immediately swap for USDC to top off its fee/P&L pool.


## pnl pool mechanics

Any account can call the settlePNL instruction, which will trigger negative P&L accounts to be settled, adding to each per-market P&L pool. Negative P&L being settled increases the amount available to be settled, whilst positive P&L being settled decreases the amount available for settlement.


***Note***: calling settlePNL does not affect open positions. The function only settles the funds available in the PNL Pool for withdrawal.

It's important to recognise the difference between **settling P&L** and **realising P&L** (read more here: P&L.

**Calling settlePNL**

Any account can call settlePNL instruction. Once called, all unrealised negative P&L will be settled and added to the market's P&L Pool to be made available for withdrawal.

Users with open positions that have negative unrealised P&L will have their unrealised P&L settled and sent to the P&L Pool; **however, their position will be unaffected**.

As users are settled against, the Cost Basis for their position will be adjusted so that their position remains unchanged even though a portion of their unrealised negative P&L has been realised and sent to the P&L Pool.

The P&L settled as a result of the settlePNL instruction will be reflected in theUnrealised P&L tab, specifically within theRealised P&L column.  The adjusted cost basis for the position is reflected in the Cost Basis column.

## fee pool mechanics
Both Perpetual and Spot Markets have Fee Pools. Fee pools are set up to accumulate a fraction of exchange fees to eventually be sent to the associated Revenue Pool.


Perpetual Market Fee Pool can be utilized for bankruptcy resolution after its insurance fund threshold is reached.


## revenue pool mechanics

A portion of the fees collected within the protocol (denominated in a particular token: USDC, SOL, etc) go into that spot market's revenue pool.

The revenue pool increases from various portions of the protocol:

1. borrow fees
2. spot market exchange fees
3. perpetual market exchange fees
4. liquidations

and ultimately goes to fund:

1. insurance vault
2. perpetual market amm (conditionally)

**Insurance Fund**

Every hour (on the hour), a portion of the revenue pool can be settled to the insurance fund using the permissionless settle_revenue_to_insurance_fund instruction:

If the insurance fund has users staked, each individual hourly settlement is capped to what would amount to 1000% APR

- thus an astronomically large inflow into the revenue pool (relatively to user insurance staked amounts) would result in revenue that slowly reaches the insurance over a longer period of time rather than immediately
    - this encourages more insurance fund stakers (who require a medium horizon of insurance offering) to join during the high annualised cap inflow

Insurance Fund Stakers must adhere to the cooldown period for withdrawals (see Insurance Fund Staking).

**Spot Markets**

Spot Markets allow for swaps between tokens and interest payments between depositors and borrowers. These token swaps and flow of interest are parameterised to allow fee collection for the revenue pool and thus ultimately insurance.

Within the program, its parameterised by the following **in bold**:

| Field | Description |
| --- | --- |
| total_if_factor | percentage of the borrow interest reserved for revenue pool |
| user_if_factor | this proportion of total_if_factor is reserved for staked users (the other piece is reserved for the protocol itself) |
| liquidation_if_factor | the proportion of liability transfer in liquidate_borrow that is sent to the revenue pool |

Thus the following must be true: user_if_factor <= total_if_factor.  For example, if the total_if_factor is 100%, depositors would receive no interest from borrows.

The following instructions interact w/ the insurance fund:

- resolve_borrow_bankruptcy

**Perpetual Markets**

Perpetual Markets are bootstrapped by the Drift AMM which depending on market-making performance conditions can add and remove funds from the revenue pool.

Within the program, its parameterized by the following **in bold**:

| Field | Description |
| --- | --- |
| max_revenue_withdraw_per_period | the amm's max revenue pool draw per period
(note this doesn't include bankruptcy resolution) |
| revenue_withdraw_since_last_settle | revenue pool draws on behalf of user pnl since the last settle
(note this doesn't include bankruptcy resolution) |
| last_revenue_withdraw_ts | the last timestamp of a revenue withdraw 
(track in order to reset the period) |

A perpetual market's amm may draw up to max_revenue_withdraw_per_period from the revenue pool every period.

Additionally, for direct draws from the insurance fund, it parameterized by the following **in bold**:

| Field | Description |
| --- | --- |
| quote_settled_insurance | settled funds from the insurance fund since inception |
| quote_max_insurance | max funds it can settle from insurance fund since inception |
| unrealized_max_imbalance | max amount of pnl the net users can be owed within a market before:
1. draws from insurance are allowed
2. initial asset weights for this pnl gets discounted |

Unlike spot markets, perp markets are capped by the max draw from insurance via quote_max_insurance

quote_settled_insurance tracks the insurance fund draw amount since inception. Once this threshold is reached or the insurance fund is depleted, the market will then resort to the AMM Fee Pool. For any remaining losses not covered, the market will perform socialized losses in bankruptcy events.

The following instructions interact w/ the insurance fund:

- resolve_perp_pnl_deficit
- resolve_perp_bankruptcy

notes:

resolve_perp_pnl_deficit can only be resolved by insurance fund deposits (within the market's constraints), not by social loss with other users


## INSURANCE FUND mechanics


**Staking**



**Reward**

For providing liquidity to the Insurance Fund, Insurance Fund Stakers are rewarded with their proportionate share of the Revenue Pool every hour.

The Revenue Pool is funded by various aspects of the protocol:

1. borrow fees;
2. spot market exchange fees;
3. perpetual market exchange fees; and
4. liquidation fees.

An Insurance Fund Staker's proportionate share is calculated by: Total Staked Amount / Total Insurance Fund

Each revenue settlement is split between Insurance Fund Stakers and a protocol-owned portion of the insurance fund.

**Example**

- *The Insurance Fund is at $5000 USDC. You decide to stake $10,000 USDC, bringing the total to $15,000 USDC.*

Your proportionate share of the Revenue Pool paid every hour for stakers would be 10000/15000 = 66.6%.

Each hour, as the revenue pool settles fees earned to the Insurance Fund, let's say $30. Half of this is reserved for the protocol, while the other half is designated for stakers, making the staker payout $15. You will receive 66.6% of the payout ($10) and the remaining Insurance Fund stakers would receive 33.3% ($5).

**Unstaking** (Cooldown Period)

There is a cooldown period of ~14 days for unstaking any collateral from the Insurance Fund.

A user first requests to unstake a specific amount (denominated in shares). During the cooldown period, the user still receives rewards and is liable for user deficit resolution. After the elapsed period, the user can then unstake. Upon unstake, a net winning during the cooldown period is forgone and split amongst the current set of stakers and while a net loss is incurred by the unstaker. This means the unstake process downside only for the unstaker, and upside only for the stakers who remain.

Additionally, during the cooldown period, if a user wishes to cancel the unstake request, they forgo any gains made during the unstake request period and their share is adjusted according. Those forgone gains are also split amongst the current stakers upon cancellation.
    ```
"""




async def gpt_page(clearing_house: DriftClient): 

    gpt_ans = ""

    ch = clearing_house
    # state = await get_state_account(ch.program)
    col1, col2, col22, col3 = st.columns([6,2,2,1])
    OPENAI_API_KEY = col22.text_input('OPENAI_API_KEY', '')
    if OPENAI_API_KEY!='' and len(OPENAI_API_KEY) > 4:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY

    user_q = col1.text_area('ask drift-gpt:')

    def do_gpt(user_q, mode='boring'):
        og_user_q = str(user_q)
        user_q = str(user_q)
        context = ""
        if 'insurance' in user_q or 'pool' in user_q or 'pnl' in user_q.lower():
            context = INSURANCE_POOL_CONTEXT
        else:
            context = DRIFT_CONTEXT

        if '?' not in user_q:
            user_q+='?'

        if mode == 'cheeky':
            user_q = 'answer tongue-in-cheek and party casual style. ' + user_q
        elif mode == 'uberzahl':
            user_q = 'answer in style of sophisticated mid-century mad-scientist. ' + user_q
        elif mode == 'boring':
            user_q = 'answer in style of a know-it-all smart contract developer. ' + user_q
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": context + user_q}]
        )

        with open('gpt_database.csv','a', newline='') as f:
            writer = csv.writer(f)
            cont = str(completion.choices[0].message.content)
            createtime = (completion.created)
            writer.writerow([og_user_q, mode, createtime, cont])

        return completion
     
    is_clicked = col2.button('submit')  
    mode = col3.radio('mode:', ['cheeky', 'uberzahl', 'boring'])
    if is_clicked and len(user_q):
        gpt_ans = do_gpt(user_q, mode) 
        is_clicked = False

    if gpt_ans:
        st.write('> ' + str(gpt_ans.choices[0].message.content))
        st.json(gpt_ans, expanded=False)



