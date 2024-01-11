from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

model = load_model('C:\\Users\\Asus\\Stock_market\\my_model.keras')

start = '2018-01-01'
end = '2023-12-31'

st.title('Stock trend prediction')
data_input = st.text_input('Enter Stock Ticker' , 'AAPL')

df = yf.download(data_input, start=start, end=end)

# Describing_data

st.subheader('Data from 2018 -2023')
st.write(df.describe())

#visulizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g')
plt.plot(ma200,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

#splitting
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)




#load model


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

#prediction
y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test *scale_factor
#vizulization

fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b', label = 'original price')
plt.plot(y_predicted, 'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('plot.png')
st.pyplot(fig2)