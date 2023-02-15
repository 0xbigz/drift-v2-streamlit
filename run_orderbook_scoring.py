import pandas as pd
from glob import glob
import numpy as np

def get_mm_score_for_snap_slot(df):
    d = df[(df.orderType=='limit') 
    # & (df.postOnly)
    ]
    d['baseAssetAmountLeft'] = d['baseAssetAmount'] - d['baseAssetAmountFilled']
    assert(len(d.snap_slot.unique())==1)

    market_index = d.marketIndex.max()
    oracle = d.oraclePrice.max()
    best_bid = d[d.direction=='long']['price'].max()
    best_ask = d[d.direction=='short']['price'].min()
    if(best_bid > best_ask):
        if best_bid > oracle:
            best_bid = best_ask
        else:
            best_ask = best_bid

    mark_price = (best_bid+best_ask)/2
    within_bps_of_price = (mark_price * .0005)

    if market_index == 1 or market_index == 2:
        within_bps_of_price = (mark_price * .00025)

    def rounded_threshold(x):
        return np.round(float(x)/within_bps_of_price) * within_bps_of_price

    d['priceRounded'] = d['price'].apply(rounded_threshold)
    # print(d)

    top6bids = d[d.direction=='long'].groupby('priceRounded').sum().sort_values('priceRounded', ascending=False)[['baseAssetAmountLeft']]
    top6asks = d[d.direction=='short'].groupby('priceRounded').sum()[['baseAssetAmountLeft']]

    tts = pd.concat([top6bids['baseAssetAmountLeft'].reset_index(drop=True), top6asks['baseAssetAmountLeft'].reset_index(drop=True)],axis=1)
    tts.columns = ['bs','as']
    # print(tts)
    min_q = (5000/mark_price)
    q = ((tts['bs']+tts['as'])/2).apply(lambda x: max(x, min_q)).max()
    # print('q=', q)
    score_scale = tts.min(axis=1)/q * 100
    score_scale = score_scale * pd.Series([2, .75, .5, .4, .3, .2]) #, .09, .08, .07])

    for i,x in enumerate(top6bids.index[:6]):
        ba = d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'baseAssetAmountLeft']
        ba /= ba.sum()
        d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'score'] = score_scale.values[i] * ba
    for i,x in enumerate(top6asks.index[:6]):
        ba = d.loc[(d.priceRounded==x)  & (d.direction=='short'), 'baseAssetAmountLeft']
        ba /= ba.sum()
        d.loc[(d.priceRounded==x) & (d.direction=='short'), 'score'] = score_scale.values[i] * ba
    
    return d


market_indexes = [0,1,2,3]
for mi in market_indexes:
    tt = 'perp'+str(mi)
    ggs = glob('../drift-v2-orderbook-snap/'+tt+'/*.csv')
    dfs = []
    for x in sorted(ggs)[-3500:]:
        df = pd.read_csv(x) 
        df['snap_slot'] = int(x.split('_')[-1].split('.')[0])
        df = get_mm_score_for_snap_slot(df)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.to_csv('data/'+tt+'.csv.gz', index=False, compression='gzip')