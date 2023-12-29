import streamlit as st
from solders.rpc.responses import GetSignaturesForAddressResp, GetTokenAccountBalanceResp, GetTransactionResp
import json
from anchorpy.provider import Signature
from solders.pubkey import Pubkey

async def load_token_balance(connection, address):
    res: GetTokenAccountBalanceResp = (await connection.get_token_account_balance(address)).to_json()
    res2 = json.loads(res)
    v_amount = int(res2['result']['value']['amount'])
    return v_amount


async def transaction_history_for_account(connection, addy, before_sig1, limit, MAX_LIMIT):

    if isinstance(addy, str):
         addy = Pubkey.from_string(addy)

    res2 = []
    first_try = True
    while (first_try and len(res2) % 1000 == 0) and len(res2)< MAX_LIMIT:
        # try:
            if len(res2):
                bbs = res2[-1]['signature']
                bbs = Signature.from_string(bbs)
            else:
                bbs = before_sig1
            res: GetSignaturesForAddressResp = (await connection.get_signatures_for_address(addy, 
                                                                                            before=bbs, 
                                                                                            limit=limit
                                                                                            )).to_json()
            res = json.loads(res)
            if 'result' not in res:
                st.warning('bad get_signatures_for_address' + str(res))
                first_try = False
            else:
                res2.extend(res['result'])
        # except Exception as e:
        #     st.warning('exception:'+str(e))
        #     first_try = False

    return res2