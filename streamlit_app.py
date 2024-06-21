# 載入必要模組
import os
import numpy as np
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib as ta
import datetime

###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

## 读取Pickle文件
@st.cache(ttl=3600, show_spinner="正在加載資料...")
def load_data(url):
    df = pd.read_pickle(url)
    return df 

df_original = load_data('kbars_2330_2022-01-01-2022-11-18.pkl')

df_original = pd.read_pickle('kbars_2330_2022-01-01-2022-11-18.pkl')
df_original = df_original.drop('Unnamed: 0', axis=1)

##### 選擇資料區間
st.subheader("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')
start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

###### (2) 轉化為字典 ######
KBar_dic = df.to_dict()
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open'] = np.array(KBar_open_list)
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list]
KBar_dic['time'] = np.array(KBar_time_list)
KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low'] = np.array(KBar_low_list)
KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high'] = np.array(KBar_high_list)
KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close'] = np.array(KBar_close_list)
KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume'] = np.array(KBar_volume_list)
KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount'] = np.array(KBar_amount_list)

######  (3) 改變 KBar 時間長度 ########
import indicator_forKBar_short

Date = start_date.strftime("%Y-%m-%d")
st.subheader("設定一根 K 棒的時間長度")
option = st.selectbox("設定一根 K 棒的時間長度", ("以分鐘為單位", "以 日 為單位", "以 週 為單位", "以 月 為單位"))
cycle_duration = st.number_input('輸入一根 K 棒的時間長度', key="KBar_duration")

cycle_duration = int(cycle_duration)
KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

for i in range(KBar_dic['time'].size):
    time = KBar_dic['time'][i]
    open_price = KBar_dic['open'][i]
    close_price = KBar_dic['close'][i]
    low_price = KBar_dic['low'][i]
    high_price = KBar_dic['high'][i]
    qty = KBar_dic['volume'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

KBar_dic = {}
KBar_dic['time'] = KBar.TAKBar['time']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] = KBar.TAKBar['high']
KBar_dic['low'] = KBar.TAKBar['low']
KBar_dic['close'] = KBar.TAKBar['close']
KBar_dic['volume'] = KBar.TAKBar['volume']

###### (4) 計算各種技術指標 ######
KBar_df = pd.DataFrame(KBar_dic)

#####  (i) 移動平均線策略   #####
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2)

KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

#####  (ii) RSI 策略   #####
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2)

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

#####  (iii) MACD 計算   #####
st.subheader("設定MACD參數")
fastperiod = st.slider('選擇快速均線週期', 0, 100, 12)
slowperiod = st.slider('選擇慢速均線週期', 0, 100, 26)
signalperiod = st.slider('選擇訊號線週期', 0, 100, 9)

KBar_df['MACD'], KBar_df['MACDSignal'], KBar_df['MACDHist'] = ta.MACD(KBar_df['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

last_nan_index_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)][0]

#####  (iv) 布林帶計算   #####
st.subheader("設定布林帶參數")
bbands_period = st.slider('選擇布林帶週期', 0, 100, 20)
bbands_stddev = st.slider('選擇布林帶標準差倍數', 0.0, 10.0, 2.0)

KBar_df['UpperBand'], KBar_df['MiddleBand'], KBar_df['LowerBand'] = ta.BBANDS(KBar_df['close'], timeperiod=bbands_period, nbdevup=bbands_stddev, nbdevdn=bbands_stddev, matype=0)

last_nan_index_BBands = KBar_df['MiddleBand'][::-1].index[KBar_df['MiddleBand'][::-1].apply(pd.isna)][0]

###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = ['日期', '商品', '開盤價', '最高價', '最低價', '收盤價', '成交量', 'MA_long', 'MA_short', 'RSI_long', 'RSI_short', 'RSI_Middle', 'MACD', 'MACDSignal', 'MACDHist', 'UpperBand', 'MiddleBand', 'LowerBand']

###### (6) 畫圖  ######

#####  (i) 畫 K 線圖  ######
KBar_df_noNaN = KBar_df.iloc[last_nan_index_MA:last_nan_index_RSI]
fig_KBar = go.Figure(data=[go.Candlestick(x=KBar_df_noNaN['日期'],
                open=KBar_df_noNaN['開盤價'],
                high=KBar_df_noNaN['最高價'],
                low=KBar_df_noNaN['最低價'],
                close=KBar_df_noNaN['收盤價'])])

#####  (ii) 畫 MA  ######
fig_KBar.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['MA_long'],
                    mode='lines',
                    name='長移動平均線'))
fig_KBar.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['MA_short'],
                    mode='lines',
                    name='短移動平均線'))

#####  (iii) 畫 RSI  ######
fig_RSI = make_subplots(specs=[[{"secondary_y": True}]])
fig_RSI.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['RSI_long'], name='長RSI', line=dict(color='blue')), secondary_y=True)
fig_RSI.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['RSI_short'], name='短RSI', line=dict(color='red')), secondary_y=True)
fig_RSI.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['RSI_Middle'], name='RSI中線', line=dict(color='grey', dash='dash')), secondary_y=True)

#####  (iv) 畫 MACD  ######
fig_MACD = make_subplots(specs=[[{"secondary_y": True}]])
fig_MACD.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['MACD'], name='MACD', line=dict(color='blue')), secondary_y=True)
fig_MACD.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['MACDSignal'], name='訊號線', line=dict(color='red')), secondary_y=True)
fig_MACD.add_trace(go.Bar(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['MACDHist'], name='MACDHist', marker_color='grey'), secondary_y=False)

#####  (v) 畫布林帶  ######
fig_KBar.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['UpperBand'], mode='lines', name='布林帶上軌', line=dict(color='green', dash='dash')))
fig_KBar.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['MiddleBand'], mode='lines', name='布林帶中線', line=dict(color='grey', dash='dash')))
fig_KBar.add_trace(go.Scatter(x=KBar_df_noNaN['日期'], y=KBar_df_noNaN['LowerBand'], mode='lines', name='布林帶下軌', line=dict(color='green', dash='dash')))

##### (vi) 展示圖表 ######
with st.expander("K線圖", expanded=True):
    st.plotly_chart(fig_KBar)

with st.expander("RSI圖表", expanded=False):
    st.plotly_chart(fig_RSI)

with st.expander("MACD圖表", expanded=False):
    st.plotly_chart(fig_MACD)
