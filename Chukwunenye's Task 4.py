#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import math


# ## loading of data

# In[2]:


data1=pd.read_csv('sales.csv')
data2=pd.read_csv('sensor_stock_levels.csv')
data3=pd.read_csv('sensor_storage_temperature.csv')


# In[3]:


data1.drop(columns=['Unnamed: 0'],axis=0,inplace=True)
data2.drop(columns=['Unnamed: 0'],axis=0,inplace=True)
data3.drop(columns=['Unnamed: 0'],axis=0,inplace=True)


# ## converting timeframe

# In[4]:


# created a function to convert timeframe into an object
def time_converter(data:pd.DataFrame= None, column: str = None):
    df=pd.to_datetime(data['timestamp'],format="%Y-%m-%d %H:%M:%S")
    data[column]=df
    
    return 


# In[5]:


time_converter(data1,'timestamp')
time_converter(data2,'timestamp')
time_converter(data3,'timestamp')


# In[6]:


# created a function to remove the minute and second from the timeframe
def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
  dummy = data.copy()
  new_ts = dummy[column]
  new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
  new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
  dummy[column] = new_ts
  return dummy


# In[7]:


data1=convert_timestamp_to_hourly(data1,'timestamp')
data2=convert_timestamp_to_hourly(data2,'timestamp')
data3=convert_timestamp_to_hourly(data3,'timestamp')


# ## merging the datasets into one

# In[8]:


salesg=data1.groupby(['timestamp','product_id']).agg({'quantity': 'sum'}).reset_index()
stockg=data2.groupby(['timestamp','product_id']).agg({'estimated_stock_pct':'mean'}).reset_index()
tempg=data3.groupby(['timestamp']).agg({'temperature':'mean'}).reset_index()


# In[9]:


data_m1=pd.merge (salesg, stockg,how='right',on=['timestamp','product_id'])
data_m2= pd.merge(data_m1,tempg, how='left',on= 'timestamp')
data_m2['quantity']=data_m2['quantity'].fillna(0)
prod_cat=data1[['product_id','unit_price','category']]
prod_cat=prod_cat.drop_duplicates()
data_m3= pd.merge(data_m2,prod_cat, on='product_id',how='left')


# ## Feature Engineering

# In[10]:


data_m3['timestamp_day_of_month'] = data_m3['timestamp'].dt.day
data_m3['timestamp_day_of_week'] = data_m3['timestamp'].dt.dayofweek
data_m3['timestamp_hour'] = data_m3['timestamp'].dt.hour
data_m3.drop(columns=['timestamp'], inplace=True)
data_m3=pd.get_dummies(data_m3,columns=['category'])
data_m3.drop(columns=['product_id'],axis=0,inplace=True)


# ## assigning the target and independent variables

# In[11]:


y=data_m3['estimated_stock_pct']
x=data_m3.drop(columns=['estimated_stock_pct'],axis=0)


# ## Machine Learning Model

# In[12]:


x_train, x_test, y_train,y_test= train_test_split(x,y,test_size=0.20, random_state=2) 
s=StandardScaler()
s.fit_transform(x_train,y_train)
x_test=s.transform(x_test)


# In[13]:


# LinearRegresion model
lr=LinearRegression()
mod=lr.fit(x_train,y_train)
pred=mod.predict(x_test)


# In[14]:


#RandomForestRegressor model
model = RandomForestRegressor()
mod2=model.fit(x_train,y_train)
pred2=mod2.predict(x_test)


# ## Accuracy Check

# In[15]:


print ('The metrics score of the LinearRegression model')
print('mse:',mean_squared_error(y_test, pred))
print('rmse:',math.sqrt(mean_squared_error(y_test, pred)))
print('mae', mean_absolute_error(y_test, pred))


# In[16]:


print('The metrics score of the RandomForestRegressor model')
print('mse:',mean_squared_error(y_test, pred2))
print('rmse:',math.sqrt(mean_squared_error(y_test, pred2)))
print('mae:',mean_absolute_error(y_test, pred2))

