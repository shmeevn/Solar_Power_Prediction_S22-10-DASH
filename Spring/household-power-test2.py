#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from statsmodels.tsa.stattools import adfuller
get_ipython().system('pip install pmdarima --quiet')
import pmdarima as pm


# In[37]:


#read text file in index
df = pd.read_csv('C:\\Users\\Faisa_k2n8tj8\\OneDrive\\Desktop\\household.txt', sep=';',
                  parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'],index_col='dt')


# In[38]:


# Check the size of our data (Number of rows and columns)

df.shape


# In[39]:


# Check sample records of our data

df.sample()


# In[40]:


# check missing values

df.isnull().sum()


# In[41]:


# fill missing values using forward-fill and make the changes permanent in the original dataframe

df.ffill(axis=0,inplace=True)


# In[42]:


# Feature Engineering - Creation of new Feature 'Power Consumption'

eq1 = (df['Global_active_power']*1000/60) 
eq2 = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
df['power_consumption'] = eq1 - eq2
df.head()


# In[43]:


# Creation of Date Time column for aggregation operation
df['Date'] = df.index.date
df['time'] = df.index.time

df['Date'] = df['Date'].astype(str)
df['time'] = df['time'].astype(str)

df.info()


# In[44]:


# Convert into DateTime and dropping from Original df

df['exact_time'] = df['Date']+";"+df['time']
df['exact_time_DT'] = pd.to_datetime(df['exact_time'],format="%Y-%m-%d;%H:%M:%S")
data = df.drop(['Date', 'time','exact_time'],axis = 1).sort_values(by=['exact_time_DT'])
data.head(10)


# In[45]:


# filter out 2006 data, only keep data post 2006
df_subset = data[data.index.year>2006]
df_subset.shape


# In[46]:


# Grouped on Date column by Week 

df_subset = df_subset.groupby(pd.Grouper(key='exact_time_DT',freq='M')).sum()

df_subset.head()


# In[47]:


# subset of only power consumption column

df_power_consumption = df_subset[['power_consumption']]


# In[48]:


# plot normal graph for power consumption, date which us year and month in x and y is power consumption been use

plt.figure(figsize=(15,7))
plt.title("Power Consumption")
plt.xlabel('Date')
plt.ylabel('power_consumption')
plt.plot(df_power_consumption['power_consumption'],label='Power_Consumption')
plt.legend(loc="best")
plt.show()


# In[49]:


#Augmented Dickey–Fuller test:

print('Results of Dickey Fuller Test:')
print('*'*50)
dftest = adfuller(df_power_consumption['power_consumption'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)


# In[50]:


#Standard ARIMA Model, which is predict future values based on past values

ARIMA_model = pm.auto_arima(df_power_consumption['power_consumption'], 
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)


# In[51]:


ARIMA_model.plot_diagnostics(figsize=(15,12))
plt.show()


# In[ ]:




