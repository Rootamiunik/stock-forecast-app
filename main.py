import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import plotly.express as px
import prophet
from prophet.plot import plot_plotly
from plotly.subplots import make_subplots
import random

#Constants
START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(
    layout = 'wide',
    page_title = 'Stock Forecast App'
)
st.title("Stock Forecast App")

with st.sidebar:
    st.image("logo.png",caption="Stock Forecast App")
    st.header("Parameters")
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','TSLA')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365
    
# data loading and creating dataframe
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

df = load_data(selected_stock)
st.write(df.tail())
	
#coloums creation 
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with col2: 
    new_fig = px.bar(df,x='High',y="Low",color="High",color_continuous_scale="reds")
    new_fig.layout.update(title_text="High vs Low",xaxis_rangeslider_visible=True)
    st.plotly_chart(new_fig)

df_train = df[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = prophet.Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

fig1= go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Actual_stock"))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name="Predicted_stock"))
fig1.layout.update(title_text=f'Forecast plot for {n_years} years', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.subheader("Forecast Component")
col3,col4 = st.columns(2)

with col3:
    fig2 = px.bar(forecast,x="ds",y="trend",color="trend",color_continuous_scale="greens")
    fig2.layout.update(title_text=f"Overall trends")
    st.plotly_chart(fig2)
with col4:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=forecast['trend'], y=forecast['weekly'], name="Actual_stock"))
    fig3.layout.update(title_text=f"weekly data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

col5,col6 = st.columns(2)

df_for = forecast[["ds","yearly"]]
st.area_chart(df_for,x="ds",y="yearly")
