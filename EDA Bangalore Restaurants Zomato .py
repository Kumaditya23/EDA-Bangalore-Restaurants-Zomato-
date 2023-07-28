#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Banglore-based Restaurants

# In[1]:


# Import library:-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

pd.pandas.set_option('display.max_columns',None)


# # Reading Data 

# The dataset can be downloaded from this link:-
# https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants    

# In[3]:


dataset = pd.read_csv("E:\zomato\zomato.csv")
dataset.head()


# ### Dropping Unnecessary Columns

# In[5]:


dataset = dataset.drop(['url','address','phone','dish_liked','reviews_list','menu_item'],axis=1)
dataset.head()


# # Renaming Columns

# just making it simple to use

# In[9]:


dataset = dataset.rename(columns={'approx_cost(for two people)':'cost_for_2','listed_in(type)':'type','listed_in(city)':'city'})
dataset.head()


# # Checking Null Values

# can be done by .info() or .isnull()

# In[10]:


dataset.info()


# # Dropping Duplicates

# In[11]:


dataset.drop_duplicates(inplace=True)
dataset.info()


# # Checking Numerical features

# In[12]:


li=['rate','votes','cost_for_2']
for i in li:
    print(dataset[i].unique())


# # Cleaning Rate Feature

# It has /and new in the data so making it as numerical value.

# In[25]:


def rate(value):
    if(value=='NEW' or value=='-'):
        return np.nan
    else:
        value=str(value).split('/')
        value=value[0]
        return float(value)
dataset['rate']=dataset['rate'].apply(rate)

dataset['rate'].unique()


# # Cleaning Cost Feature

# It has , in the data

# In[28]:


def cost(value):
    value=str(value)
    if ',' in value:
        value1=value.replace(',','')
        return float(value1)
    else:
        return float(value)
    
dataset ['cost_for_2'] = dataset['cost_for_2'].apply(cost)


dataset ['cost_for_2'].unique()


# # Perfect Numerical Features with nan

# In[30]:


numerical_features=[feature for feature in dataset.columns if dataset[feature].dtype!='O']
numerical_features


# # Filling Null Values with Median.

# In[32]:


for feature in numerical_features:
    median=dataset[feature].median()
    dataset[feature].fillna(median,inplace=True)
    
dataset[numerical_features]


# Just confirming that it doesn't contain null values

# In[33]:


dataset[numerical_features].info()


# # Categorical Features

# In[36]:


cat_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']
cat_features


# # Checking Categorical features Unique Values

# In[40]:


for feature in cat_features:
    if feature !='name':
        print(dataset[feature].value_counts())
dataset[cat_features]


# # Cleaning Cat Features

# Here online_order , book_table,city have less unique values and they are perfect to visualize whereas , other features need to cleaned. so, if any unique value less than 0.5% or 1% weigtage then i am considering it as rare_variable or simply as others.

# In[46]:


cat_f=['cuisines','location','rest_type']

for feature in cat_f:
    temp=dataset[feature].value_counts()/len(dataset)
    index=temp[temp>0.005].index
    dataset[feature]=np.where(dataset[feature].isin(index),dataset[feature],'others')
    
    for feature in cat_f:
        print(dataset[feature].value_counts())


# When it comes to a place category we have two features (location,city). Location is more specific than city feature so, i am droppinh city.

# In[48]:


dataset.drop(['city'],axis=1,inplace=True)
dataset.head()


# Data is clean and perfect for visualizing

# # Visualizing Data

# ### Restaurants according to location

# From this graph , one can analyze which place is good for opening a restaurant.

# In[49]:


plt.figure(figsize=(10,5),dpi=150)
sns.countplot(x='location', data = dataset)
plt.xticks(rotation=90)
plt.savefig('count(rest)-location.jpg',bbox_inches='tight')


# ### Rating according to type of restaurant

# In[50]:


plt.figure(figsize=(10,5),dpi = 100)
sns.boxplot(x='type',y='rate',data=dataset)

plt.savefig('rate-type.jpg')


# # Booking table Option

# From this graph one can know whether to give option for booking or not

# In[51]:


plt.figure(figsize=(5,5),dpi=100)
sns.boxplot(x='book_table',y='rate',data=dataset)
plt.savefig('tablebooking.jpg')


# ### Rating according to a Restaurant type

# From this graph, One can get to know in which type people are more satisfied and in which type people are less satisfied and interested.

# In[53]:


plt.figure(figsize=(10,5),dpi=100)
sns.boxplot(x='rest_type',y ='rate',data=dataset)
plt.xticks(rotation=90)
plt.savefig('rate-rest_type.jpg',bbox_inches='tight')


# ### Votes according to thd cusines

# here others option is dominating and not giving perfect answer. so without others need to plotted.

# In[54]:


plt.figure(figsize=(10,5),dpi=100)
dataset.groupby(['cuisines'])['votes'].sum().plot.bar()
plt.show()


# In[55]:


df=dataset.groupby(['cuisines'])['votes'].sum()
df=df.to_frame()
df=df.sort_values('votes',ascending=False)
df.drop('others',axis=0,inplace = True)


# In[56]:


plt.figure(figsize=(10,5),dpi=100)
sns.barplot(df.index,df['votes'])
plt.xticks(rotation=90)
plt.savefig('votes-cuisines.jpg',bbox_inches='tight')


# ### Online order availabilty at diff Locations

# From this graph , one can know at which locations there is ordering facility more and in which location it is less and can conclude to put the facility or not

# In[57]:


df2=dataset.groupby(['location','online_order'])['name'].count()
df2=df2.to_frame()
df3=df2.pivot_table(index='location',columns='online_order')
df3


# In[58]:


df3.plot.bar(figsize=(15,8))
plt.savefig('Online_order.jpg',bbox_inches='tight')


# # No of types of restaurants according to location 

# In[59]:


df4=dataset.groupby(['location','type'])['name'].count()
df4=df4.to_frame()
df5=df4.pivot_table(index='location',columns='type',fill_value=0)
df5


# In[60]:


df5.plot.bar(figsize=(20,10))
plt.savefig('type-location.jpg',bbox_inches='tight')


# In[61]:


dataset.head()


# # Cost Vs Rating

# From this graph , one can know at which price range people rated more and more satisfied 

# In[62]:


plt.figure(figsize=(20,10),dpi=100)
sns.boxplot(x='cost_for_2',y='rate',data=dataset)
plt.xticks(rotation=90)
plt.savefig('rate-cost.jpg')


# # Leading Franchises according to count

# In[63]:


plt.figure(figsize=(15,8),dpi=100)
dff=dataset['name'].value_counts()[:25]
sns.barplot(x=dff.index,y=dff)
plt.xticks(rotation=90)
plt.savefig('Branches-count.jpg',bbox_inches='tight')


# In[ ]:




