import datetime
import pandas as pd
pd.options.plotting.backend = "plotly"
import streamlit as st
import numpy as np
import pytz

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
       'date']).reset_index(drop=True)
    oracle_series = df1.groupby('ts')['oraclePrice'].last()
    
    df1['markPrice'] = df1['quoteAssetAmountFilled']/df1['baseAssetAmountFilled']
    df1['buyPrice'] = np.nan
    df1['sellPrice'] = np.nan
    df1['buyPrice'] = df1.loc[df1[df1['takerOrderDirection']=='long'].index, 'markPrice']
    df1['sellPrice'] = df1.loc[df1['takerOrderDirection']=='short', 'markPrice']

    df1['takerPremium'] = (df1['markPrice'] - df1['oraclePrice']) * (2*(df1['takerOrderDirection']=='long')-1)
    df1['takerPremiumDollar'] = df1['takerPremium']*df1['baseAssetAmountFilled']
    df1['takerPremiumNextMinute'] = (df1['markPrice'] - df1['ts'].apply(lambda x: oracle_series.loc[x+60:].head(1).max())) * (2*(df1['takerOrderDirection']=='long')-1)

    return df1

def load_s3_data(markets, START=None, END=None):
    tzInfo = pytz.timezone('UTC')
    # markets = ['SOL', 'SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'APT-PERP']
    url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'


    assert(START >= '2022-11-04')
    if START is None:
        START = (datetime.datetime.now(tzInfo)-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if END is None:
        END = (datetime.datetime.now(tzInfo)+datetime.timedelta(days=1)).strftime("%Y-%m-%d")
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



@st.experimental_memo
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def trade_flow_analysis():
    tzInfo = pytz.timezone('UTC')

    col1, col2, col3 = st.columns(3)
    market = col1.selectbox('select market:', ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'APT-PERP', 'SOL'])

    date = col2.date_input('select date:', min_value=datetime.datetime(2022,11,4), max_value=(datetime.datetime.now(tzInfo)))
    markets_data = load_s3_data([market], date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))


    csv = convert_df(markets_data[market])

    col3.download_button(
        label="Download CSV",
        data=csv,
        file_name='drift-v2-'+market+'-'+date.strftime('%Y-%m-%d')+'.csv',
        mime='text/csv',
    )

    solperp = dedupdf(markets_data, market)

    zol1, zol2, zol3 = st.columns(3)

    takerfee = solperp['takerFee'].sum()
    makerfee = solperp['makerFee'].sum()
    fillerfee = solperp['fillerReward'].sum()

    zol1.metric(market+' Volume:', '$'+f"{(solperp['quoteAssetAmountFilled'].sum().round(2)):,}")
    zol2.metric(market+' Fees:', '$'+str((takerfee).round(2)), '$'+str(-makerfee.round(2))+' in liq/maker rebates, '+'$'+str(fillerfee.round(2))+' in filler rewards')

    showprem = zol3.radio('show premiums in plot', [True, False], 1)
    showmarkoutfees = zol3.radio('show spread vs oracle w/ markout', [True, False], 1)

    pol1, pol2 = st.columns(2)
    trade_df = solperp.set_index('date').sort_index()[['buyPrice', 'oraclePrice', 'sellPrice']]
    if showprem:
        trade_df['buyPremium'] = (trade_df['buyPrice']-trade_df['oraclePrice']).ffill()
        trade_df['sellPremium'] = (trade_df['oraclePrice']-trade_df['sellPrice']).ffill()
    fig = trade_df.plot().update_layout(
        title='fills vs oracle',
    xaxis_title="date", yaxis_title="price"
)

    pol1.plotly_chart(fig)


    retail_takers = solperp[solperp.takerOrderId < 10000]['taker'].unique()
    bot_takers = solperp[solperp.takerOrderId >= 10000]['taker'].unique()

    if showmarkoutfees:
        solperp['markout_pnl_1MIN'] = -(solperp['baseAssetAmountFilled'] * (solperp['takerPremiumNextMinute']))
    else:
        solperp['markout_pnl_1MIN'] = -(solperp['baseAssetAmountFilled'] * (solperp['takerPremiumNextMinute'] - solperp['takerPremium']))
    markoutdf = pd.concat({
        'retailMarkout1MIN:': solperp[solperp.taker.isin(retail_takers)].set_index('date').sort_index()['markout_pnl_1MIN'].resample('1MIN').last(),
     'botMarkout1MIN': solperp[solperp.taker.isin(bot_takers)].set_index('date').sort_index()['markout_pnl_1MIN'].resample('1MIN').last(),
    },axis=1).cumsum().ffill()
    fig2 = markoutdf.plot().update_layout(title='markout',
    xaxis_title="date", yaxis_title="PnL (USD)"
)
    pol2.plotly_chart(fig2)


    def show_analysis_tables(nom, takers):
        def make_clickable(link):
            # target _blank to open new window
            # extract clickable text to display for your link
            text = link.split('=')[1]
            ts = text[:4]+'...'+text[-4:]
            return f'<a target="_blank" href="{link}">{ts}</a>'
        st.write(nom, len(takers))
        with st.expander('users'):
            tt = pd.DataFrame(solperp[solperp.taker.isin(takers)].groupby('taker')[['takerFee', 'quoteAssetAmountFilled', 'takerPremium', 'takerPremiumNextMinute', 'takerPremiumDollar']]\
                .agg({'takerFee':'count', 'quoteAssetAmountFilled':'sum', 'takerPremium':'sum', 'takerPremiumNextMinute':'sum', 'takerPremiumDollar':'sum' })\
                .sort_values(by='quoteAssetAmountFilled', ascending=False))
            tt.columns = ['count', 'volume', 'takerPremiumTotal', 'takerPremiumNextMinuteTotal', 'takerUSDPremiumSum',]
            tt['Markout'] = -(tt['takerPremiumNextMinuteTotal'] - tt['takerPremiumTotal'])
            tt['userAccount'] =  ['https://app.drift.trade?userAccount='+str(x) for x in tt.index]
            tt['userAccount'] =  tt['userAccount'].apply(make_clickable)
            zol1, zol2 = st.columns([1,4])
            zol2.dataframe(tt.drop(['userAccount'], axis=1), height=len(tt)*40, use_container_width=True)
            zol1.write(tt[['userAccount', 'volume']].reset_index(drop=True).to_html(index=False, escape=False), unsafe_allow_html=True)

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
        df1['Markout'] = -(df1['takerPricePremiumNextMinuteMean'] - df1['takerPricePremiumMean'])

        df2 = solperp[solperp.taker.isin(takers)].groupby('actionExplanation').agg({'fillerReward':'count', 
        'quoteAssetAmountFilled':'sum', 'takerPremium':'mean', 'takerPremiumNextMinute':'mean', 'takerPremiumDollar':'sum', 
        
         'takerFee':'sum',
                                                                                        'makerFee':'sum',
        })  
        df2.columns = ['count', 'volume', 'takerPricePremiumMean', 'takerPricePremiumNextMinuteMean', 'takerUSDPremiumSum',
                        'takerFeeSum', 'makerFeeSum', 
        ]
        df2['Markout'] = -(df2['takerPricePremiumNextMinuteMean'] - df2['takerPricePremiumMean'])


        st.dataframe(df1)
        st.dataframe(df2)      

    show_analysis_tables('retail takers:', retail_takers)      
    show_analysis_tables('bot takers:', bot_takers)                                                                                                            