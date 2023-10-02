import streamlit as st

async def transaction_history_for_account(connection, addy, before_sig1, limit, MAX_LIMIT):
    res2 = []
    first_try = True
    while (first_try and len(res2) % 1000 == 0) and len(res2)< MAX_LIMIT:
        try:
            if len(res2):
                bbs = res2[-1]['signature']
            else:
                bbs = before_sig1
            res = await connection.get_signatures_for_address(addy, before=bbs, limit=limit)
            if 'result' not in res:
                st.warning('bad get_signatures_for_address' + str(res))
                first_try = False
            else:
                res2.extend(res['result'])
        except Exception as e:
            st.warning('exception:'+str(e))
            first_try = False

    return res2