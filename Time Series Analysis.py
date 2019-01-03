
# coding: utf-8

# In[71]:


#This notebook is just a reference to learn more about timeseries and the different functionalities possible
#Reference Everything you can do with a timeseries Siddhart yadav


# In[210]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight') 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[169]:


google = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
google.head()


# In[20]:


humidity = pd.read_csv('humidity.csv', index_col='datetime', parse_dates=['datetime'])
humidity.head()


# In[11]:


#Google stocks don't have null data but humidity does have
#Let us remove the NA


# In[23]:


humidity = humidity.iloc[1:]
humidity = humidity.fillna(method='ffill')
humidity.head()


# In[24]:


#Let us visualize it


# In[29]:


humidity["Kansas City"].asfreq('M').plot()


# In[30]:


#Plotting monthly humidy levels of Kancas city


# In[38]:


google['2008':'2011'].plot(subplots = 'T', figsize = (10,12))
plt.title('Google stock from 2008 to 2011')
plt.show()


# In[39]:


#Creating Time stamps


# In[40]:


timestamp = pd.Timestamp(2017, 1, 1, 12)
timestamp


# In[41]:


period = pd.Period('2017-01-01')
period


# In[42]:


period.start_time < timestamp < period.end_time


# In[43]:


new_period = timestamp.to_period(freq='H')
new_period


# In[46]:


new_timestamp = period.to_timestamp(freq='H', how='start')
new_timestamp


# In[47]:


#Using daterange


# In[48]:


dr1 = pd.date_range(start='1/1/18', end='1/9/18')
dr1


# In[51]:


dr2 = pd.date_range(start='1/1/18', end='1/1/19', freq='M')
dr2


# In[52]:


# Creating a datetimeindex without specifying start date and using periods
dr3 = pd.date_range(end='1/4/2014', periods=8)
dr3


# In[59]:


# Creating a datetimeindex specifying start date , end date and periods
dr4 = pd.date_range(start='2013-04-24', end='2014-11-27', periods=3)
dr4


# In[61]:


#Using to_datetime


# In[62]:


df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
df


# In[63]:


df = pd.to_datetime(df)
df


# In[64]:


df = pd.to_datetime('01-01-2017')
df


# In[65]:


#Shifting and lags


# In[69]:


humidity["Vancouver"].asfreq('M').plot(legend=True)
shifted = humidity["Vancouver"].asfreq('M').shift(10).plot(legend=True)
shifted.legend(['Vancouver','Vancouver_lagged'])
plt.show()


# In[70]:


#Resampling


# In[90]:


# Let us use pressure data to demonstrate this
pressure = pd.read_csv('pressure.csv', index_col='datetime', parse_dates=['datetime'])
pressure.tail()


# In[91]:


pressure = pressure.iloc[1:]
pressure = pressure.fillna(method='ffill')
pressure.head()


# In[92]:


pressure = pressure.fillna(method='bfill')
pressure.head()


# In[93]:


pressure.shape


# In[94]:


# We downsample from hourly to 3 day frequency aggregated using mean
pressure = pressure.resample('3D').mean()
pressure.head()


# In[95]:


pressure.shape


# In[96]:


pressure = pressure.resample('D').pad()
pressure.head()


# In[97]:


# Shape after resampling(upsampling)
pressure.shape


# In[98]:


#Finance and statistics


# In[ ]:


google['Change'] = google.High.div(google.High.shift())
google['Change'].plot(figsize=(20,8))


# In[101]:


google['High'].plot()


# In[115]:


#Percent change


# In[117]:


google['Change'] = google.High.div(google.High.shift())
google['Change'].plot(figsize=(20,8))


# In[119]:


#Stock Returns


# In[120]:


google['Return'] = google.Change.sub(1).mul(100)
google['Return'].plot(figsize=(20,8))


# In[123]:


google.High.pct_change().mul(100).plot(figsize=(20,6)) 


# In[126]:


google.High.diff().plot(figsize=(20,6))


# In[127]:


#Comparing time series


# In[128]:


#Let us compare google and microsoft


# In[129]:


microsoft = pd.read_csv('MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])


# In[130]:


# Plotting before normalization
google.High.plot()
microsoft.High.plot()
plt.legend(['Google','Microsoft'])
plt.show()


# In[134]:


#We will compare 2 time series by normalizing them. 
#This is achieved by dividing each time series element of all time series by the first element. 
#This way both series start at the same point and can be easily compared.


# In[140]:


# Normalizing and comparison
# Both stocks start from 100
normalized_google = google.High.div(google.High.iloc[0]).mul(100)
normalized_microsoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)
normalized_google.plot()
normalized_microsoft.plot()
plt.legend(['Google','Microsoft'])
plt.show()


# In[141]:


#Window functions


# In[142]:


# Rolling window functions
rolling_google = google.High.rolling('90D').mean()
google.High.plot()
rolling_google.plot()
plt.legend(['High','Rolling Mean'])
# Plotting a rolling mean of 90 day window with original High attribute of google stocks
plt.show()


# In[150]:


# Expanding window functions
microsoft_mean = microsoft.High.expanding().mean()
microsoft_std = microsoft.High.expanding().std()
microsoft.High.plot()
microsoft_mean.plot()
microsoft_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.show()


# In[158]:


#OHLC charts


# In[172]:


import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from pylab import rcParams
from plotly import tools


# In[173]:


# OHLC chart of June 2008
trace = go.Ohlc(x=google['06-2008'].index,
                open=google['06-2008'].Open,
                high=google['06-2008'].High,
                low=google['06-2008'].Low,
                close=google['06-2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[174]:


# OHLC chart of 2008
trace = go.Ohlc(x=google['2008'].index,
                open=google['2008'].Open,
                high=google['2008'].High,
                low=google['2008'].Low,
                close=google['2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[171]:


trace = go.Ohlc(x=google.index,
                open=google.Open,
                high=google.High,
                low=google.Low,
                close=google.Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[175]:


#Candlestick charts


# In[176]:


# Candlestick chart of march 2008
trace = go.Candlestick(x=google['03-2008'].index,
                open=google['03-2008'].Open,
                high=google['03-2008'].High,
                low=google['03-2008'].Low,
                close=google['03-2008'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[177]:


trace = go.Candlestick(x=google.index,
                open=google.Open,
                high=google.High,
                low=google.Low,
                close=google.Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[178]:


#ACF and #PACF
#ACF measures how a series is correlated with itself at different lags.


# In[181]:


#PACF
#The partial autocorrelation function can be interpreted as a regression of the series against its past lags.


# In[226]:


import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error


# In[183]:


# Autocorrelation of humidity of San Diego
plot_acf(humidity["San Diego"],lags=25,title="San Diego")
plt.show()


# In[184]:


# Partial Autocorrelation of humidity of San Diego
plot_pacf(humidity["San Diego"],lags=25)
plt.show()


# In[185]:


# Partial Autocorrelation of closing price of microsoft stocks
plot_pacf(microsoft["Close"],lags=25)
plt.show()


# In[186]:


#Here, only 0th, 1st and 20th lag are statistically significant.


# In[187]:


#Time series decomposition


# In[188]:


# Let's take Google stocks High for this
google["High"].plot(figsize=(16,8))


# In[219]:


rcParams['figure.figsize'] = 9, 7
decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"],freq=360) # The frequncy is annual
figure = decomposed_google_volume.plot()
plt.show()


# In[197]:


#White noise


# In[200]:


# Plotting white noise
rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc is mean, scale is variance
plt.plot(white_noise)


# In[201]:


# Plotting autocorrelation of white noise
plot_acf(white_noise,lags=20)
plt.show()


# In[204]:


# Augmented Dickey-Fuller test on volume of google and microsoft stocks 
adf = adfuller(microsoft["Volume"])
print("p-value of microsoft: {}".format(float(adf[1])))
adf = adfuller(google["Volume"])
print("p-value of google: {}".format(float(adf[1])))


# In[209]:


seed(9)
rcParams['figure.figsize'] = 16, 6
random_walk = normal(loc=0, scale=0.01, size=1000)
plt.plot(random_walk)
plt.show()


# In[211]:


sns.distplot(random_walk)


# In[212]:


#Stationarity


# In[220]:


# The original non-stationary plot
decomposed_google_volume.trend.plot()


# In[221]:


# The new stationary plot
decomposed_google_volume.trend.diff().plot()


# In[222]:


#AR models


# In[223]:


#Simulating AR1 model


# In[230]:


# AR(1) MA(1) model:AR parameter = +0.9
rcParams['figure.figsize'] = 16, 12
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)

# We will take care of MA model later
# AR(1) MA(1) AR parameter = -0.9
plt.subplot(4,1,2)
ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma2 = np.array([1])
AR2 = ArmaProcess(ar2, ma2)
sim2 = AR2.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = -0.9')
plt.plot(sim2)

# AR(2) MA(1) AR parameter = 0.9
plt.subplot(4,1,3)
ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma3 = np.array([1])
AR3 = ArmaProcess(ar3, ma3)
sim3 = AR3.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = +0.9')
plt.plot(sim3)

# AR(2) MA(1) AR parameter = -0.9
plt.subplot(4,1,4)
ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma4 = np.array([1])
AR4 = ArmaProcess(ar4, ma4)
sim4 = AR4.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = -0.9')
plt.plot(sim4)
plt.show()


# In[231]:


#Forecasting a simulated model


# In[235]:


model = ARMA(sim1, order=(1,0))
result = model.fit()
print(result.summary())
print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))


# In[236]:


# Predicting simulated AR(1) model 
result.plot_predict(start=900, end=1010)
plt.show()


# In[239]:


rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))


# In[242]:


# Predicting humidity level of Montreal
humid = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(1,0))
res = humid.fit()
res.plot_predict(start=1000, end=1100)
plt.show()


# In[243]:


rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[900:1000].values, result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))


# In[245]:


#That is huge quite an error


# In[247]:


# Predicting closing prices of google
gle = ARMA(google["Close"].diff().iloc[1:].values, order=(1,0))
res = gle.fit()
res.plot_predict(start=900, end=1010)
plt.show()


# In[249]:


#MA models


# In[250]:


rcParams['figure.figsize'] = 16, 6
ar1 = np.array([1])
ma1 = np.array([1, -0.5])
MA1 = ArmaProcess(ar1, ma1)
sim1 = MA1.generate_sample(nsample=1000)
plt.plot(sim1)


# In[253]:


model = ARMA(sim1, order=(0,1))
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))


# In[261]:


# Forecasting and predicting montreal humidity
model = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(0,3))
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))
result.plot_predict(start=1000, end=1100)
plt.show()


# In[262]:


rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))


# In[263]:


#ARMA models


# In[264]:


model = ARMA(microsoft["Volume"].diff().iloc[1:].values, order=(3,3))
result = model.fit()
print(result.summary())
print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))
result.plot_predict(start=1000, end=1100)
plt.show()


# In[267]:


rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))


# In[268]:


#ARIMA models


# In[269]:


rcParams['figure.figsize'] = 16, 6
model = ARIMA(microsoft["Volume"].diff().iloc[1:].values, order=(2,1,0))
result = model.fit()
print(result.summary())
result.plot_predict(start=700, end=1000)
plt.show()


# In[270]:


rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[700:1001].values, result.predict(start=700,end=1000)))
print("The root mean squared error is {}.".format(rmse))

