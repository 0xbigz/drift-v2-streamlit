import pandas as pd
import datetime
import pytz 
import streamlit as st

def load_if_s3_data(current_year, current_month, dd, is_devnet=False):
    url_market_pp = 'https://drift-historical-data.s3.eu-west-1' if not is_devnet else 'https://drift-historical-data.s3.us-east-1'
    url_market_prefix = url_market_pp+'.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'
    name_rot = {}
    spot_asts = ['USDC', 'SOL', 'mSOL', 
                 'wBTC', 'wETH',
                   'USDT', 'jitoSOL']
    
    if dd == 'all time':
        for name in spot_asts:
            rots = []
            for year in ['2022', '2023', str(current_year)]:
                full_if_url = url_market_prefix+name+"/insurance-fund-records/"+str(year)+"/"#+str(current_month)
                if year == '2022':
                    mrange = ['11','12']
                else:
                    mrange = range(1, current_month+1)
                for x in mrange:
                    try:
                        rot = pd.read_csv(full_if_url+str(x))
                        rot = rot.set_index('ts')
                        rot.index = pd.to_datetime((rot.index * 1e9).astype(int))
                        rots.append(rot)
                    except:
                        pass
                if len(rots):
                    rot = pd.concat(rots)
                    name_rot[name] = rot
    else:
        for name in spot_asts:
            year = str(current_year)
            full_if_url = url_market_prefix+name+"/insurance-fund-records/"+str(year)+"/"#+str(current_month)
            rots = []
            x = current_month
            try:
                rot = pd.read_csv(full_if_url+str(x))
                rot = rot.set_index('ts')
                rot.index = pd.to_datetime((rot.index * 1e9).astype(int))
                rots.append(rot)
            except:
                pass
            if len(rots):
                rot = pd.concat(rots)
                name_rot[name] = rot
            else:
                st.warning(name+ ': '+ full_if_url)
    return name_rot, spot_asts

def load_user_settlepnl(dates, user_key, with_urls=False):
    url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url += 'user/%s/settlePnls/%s/%s'
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, _) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (user_key, year, month)
        if data_url not in data_urls:
            data_urls.append(data_url)
            try:
                dd = pd.read_csv(data_url)
                dfs.append(dd)
            except:
                pass
    if len(dfs):
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame()

    if with_urls:
        return dfs, data_urls

    return dfs


def load_s3_trades_data(markets, START=None, END=None):
    tzInfo = pytz.timezone('UTC')
    # markets = ['SOL', 'SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'APT-PERP']
    # url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'

    url = 'https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'
    assert(START >= '2022-11-04')
    if START is None:
        START = (datetime.datetime.now(tzInfo)-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if END is None:
        END = (datetime.datetime.now(tzInfo)+datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    dates = [x.strftime("%Y%m%d") for x in pd.date_range(
                                                            # '2022-11-05', 
                                                        START,
                                                        END
                                                        )
            ]
    all_markets = {}

    for market in markets:
        df1 = []
        for date in dates:
            date1 = date.replace('/0', '/')
            ff = url+market+'/tradeRecords/%s/%s' % (str(date1[:4]), str(date1))
            try:
                dd = pd.read_csv(ff)
                df1.append(dd)
            except Exception as e:
                # st.warning(f'failed({str(e)}):  {market} {date} -> {date1}')
                st.warning(f'failed: {ff}')
                pass
        all_markets[market] = pd.concat(df1)
    return all_markets


def load_user_lp(dates, user_key, with_urls=False, is_devnet=False):
    url_market_pp = 'https://drift-historical-data.s3.eu-west-1' if not is_devnet else 'https://drift-historical-data.s3.us-east-1'
    url_og = url_market_pp+'.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url = url_og + 'user/%s/lp-records/%s/%s'

    dfs = []
    data_urls = []
    for date in dates:
        (year, month, _) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (user_key, year, month)
        if data_url not in data_urls:
            data_urls.append(data_url)
            try:
                dfs.append(pd.read_csv(data_url))
            except:
                pass
    if len(dfs):
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame()
    if with_urls:
        return dfs, data_urls

    return dfs

def load_volumes(dates, market_name, with_urls=False, is_devnet=False):
    url_market_pp = 'https://drift-historical-data.s3.eu-west-1' if not is_devnet else 'https://drift-historical-data.s3.us-east-1'
    url = url_market_pp+".amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/"
    url += "market/%s/trades/%s/%s/%s"
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, day) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (market_name, year, month, day)
        data_urls.append(data_url)
        try:
            dfs.append(pd.read_csv(data_url))
        except:
            pass
    dfs = pd.concat(dfs)

    if with_urls:
        return dfs, data_urls

    return dfs


def load_user_trades(dates, user_key, with_urls=False):
    url_og = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url = url_og + 'user/%s/trades/%s/%s'

    liq_url = url_og + 'user/%s/liquidations/%s/%s'

    dfs = []
    data_urls = []
    for date in dates:
        (year, month, _) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (user_key, year, month)
        if data_url not in data_urls:
            data_urls.append(data_url)
            try:
                dd = pd.read_csv(data_url, nrows=50000)
                if 'liquidation' in dd['actionExplanation'].unique():
                    if user_key in dd[dd['actionExplanation']=='liquidation'].taker.unique():
                        data_liq_url = liq_url % (user_key, year, month)
                        print(data_liq_url)
                        data_urls.append(data_liq_url)
                        dd1 = pd.read_csv(data_liq_url)
                        dd = dd.merge(dd1, suffixes=('', '_l'), how='outer', on='txSig')
                dfs.append(dd)
            except Exception as e:
                st.warning(data_url+ ' + liq files failed to load ('+str(e)+')')
    if len(dfs):
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame()
    if with_urls:
        return dfs, data_urls

    return dfs

    