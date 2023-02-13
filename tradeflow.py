import datetime
import pandas as pd
pd.options.plotting.backend = "plotly"
import streamlit as st


def dedupdf(all_markets, market_name):
    df1 = all_markets[market_name].copy()    
    df1['date'] = pd.to_datetime(df1['ts']*1e9)
    df1 = df1.drop_duplicates(['fillerReward', 'baseAssetAmountFilled', 'quoteAssetAmountFilled',
       'takerPnl', 'makerPnl', 'takerFee', 'makerRebate', 'refereeDiscount',
       'quoteAssetAmountSurplus', 'takerOrderBaseAssetAmount',
       'takerOrderCumulativeBaseAssetAmountFilled',
       'takerOrderCumulativeQuoteAssetAmountFilled', 'takerOrderFee',
       'makerOrderBaseAssetAmount',
       'makerOrderCumulativeBaseAssetAmountFilled',
       'makerOrderCumulativeQuoteAssetAmountFilled', 'makerOrderFee',
       'oraclePrice', 'makerFee', 'txSig', 'slot', 'ts', 'action',
       'actionExplanation', 'marketIndex', 'marketType', 'filler',
       'fillRecordId', 'taker', 'takerOrderId', 'takerOrderDirection', 'maker',
       'makerOrderId', 'makerOrderDirection', 'spotFulfillmentMethodFee',
       'date'])
    oracle_series = df1.groupby('ts')['oraclePrice'].last()
    
    df1['markPrice'] = df1['quoteAssetAmountFilled']/df1['baseAssetAmountFilled']
    df1['takerPremium'] = (df1['markPrice'] - df1['oraclePrice']) * (2*(df1['takerOrderDirection']=='long')-1)
    df1['takerPremiumDollar'] = df1['takerPremium']*df1['baseAssetAmountFilled']
    df1['takerPremiumNextMinute'] = (df1['markPrice'] - df1['ts'].apply(lambda x: oracle_series.loc[x+60:].head(1).max())) * (2*(df1['takerOrderDirection']=='long')-1)

    return df1

def load_s3_data(markets, START=None, END=None):
    # markets = ['SOL', 'SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'APT-PERP']
    url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'


    assert(START > '2022-11-05')
    if START is None:
        START = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if END is None:
        END = (datetime.datetime.now()+datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    dates = [x.strftime("%Y/%m/%d") for x in pd.date_range(
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
            try:
                dd = pd.read_csv(url+market+'/trades/%s' % str(date1))
                df1.append(dd)
            except:
                print('failed:', market, date, '->', date1)
                pass
        all_markets[market] = pd.concat(df1)
    return all_markets


def trade_flow_analysis():

    col1, col2 = st.columns(2)
    market = col1.selectbox('select market:', ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'APT-PERP', 'SOL'])

    date = col2.date_input('select date:', min_value=datetime.datetime(2022,11,5), max_value=(datetime.datetime.now()+datetime.timedelta(days=1)))
    markets_data = load_s3_data([market], date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))
    solperp = dedupdf(markets_data, market)

    retail_takers = solperp[solperp.takerOrderId < 10000]['taker'].unique()
    bot_takers = solperp[solperp.takerOrderId >= 10000]['taker'].unique()

    def show_analysis_tables(nom, takers):
        def make_clickable(link):
            # target _blank to open new window
            # extract clickable text to display for your link
            text = link.split('=')[1]
            return f'<a target="_blank" href="{link}">{text}</a>'
        st.write(nom, len(takers))
        with st.expander('users'):
            tt = pd.DataFrame(solperp[solperp.taker.isin(takers)].groupby('taker')['quoteAssetAmountFilled'].sum().sort_values(ascending=False))
            tt['link'] =  ['https://app.drift.trade?userAccount='+str(x) for x in tt.index]
            tt['link'] =  tt['link'].apply(make_clickable)
            st.write(tt.to_html(escape=False), unsafe_allow_html=True)

        df1 = solperp[solperp.taker.isin(takers)].groupby('takerOrderDirection').agg({
            'takerOrderDirection':'count',
            'quoteAssetAmountFilled':'sum',
                                                                                        'takerPremium':'mean',
                                                                                        'takerPremiumNextMinute':'mean',
                                                                                                                                                                                'takerPremiumDollar':'sum',
                                                                                        'takerFee':'sum',
                                                                                        'makerFee':'sum',
                                                                                        
                                                                                        })
        df1.columns = ['count', 'volume', 'takerPricePremiumMean', 'takerPricePremiumNextMinuteMean', 'takerUSDPremiumSum', 
                        'takerFeeSum', 'makerFeeSum', 
        ]
        df1['TOXIC'] = df1['takerPricePremiumNextMinuteMean'] - df1['takerPricePremiumMean']

        df2 = solperp[solperp.taker.isin(takers)].groupby('actionExplanation').agg({'fillerReward':'count', 
        'quoteAssetAmountFilled':'sum', 'takerPremium':'mean', 'takerPremiumNextMinute':'mean', 'takerPremiumDollar':'sum', 
        
         'takerFee':'sum',
                                                                                        'makerFee':'sum',
        })  
        df2.columns = ['count', 'volume', 'takerPricePremiumMean', 'takerPricePremiumNextMinuteMean', 'takerUSDPremiumSum',
                        'takerFeeSum', 'makerFeeSum', 
        ]
        df2['TOXIC'] = df2['takerPricePremiumNextMinuteMean'] - df2['takerPricePremiumMean']


        st.dataframe(df1)
        st.dataframe(df2)      

    show_analysis_tables('retail takers:', retail_takers)      
    show_analysis_tables('bot takers:', bot_takers)                                                                                                            