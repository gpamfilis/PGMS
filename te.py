
# coding: utf-8
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath('__file__'))))
# In[1]:


import pandas as pd
import numpy as np


# In[ ]:


import datetime


# In[2]:


from models.hmm_seq_fit import Scoring


# In[24]:


data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
data.loc[:,'date'] = pd.to_datetime(data.date)
ix = data[data.date==datetime.datetime(2017,8,29)].index[-1]
old_data = data.loc[:ix]
new_data = pd.read_csv('/home/kasper/Desktop/final_data_soccerway.csv')


# In[25]:


df = pd.concat([old_data,new_data], ignore_index=True)


# In[27]:


del data, old_data, new_data


# In[29]:


df.loc[:,'date'] = pd.to_datetime(df.date)


# In[30]:


input_date = datetime.datetime(2018,1,31)


# In[31]:


input_data = df[df.date==input_date]


# In[34]:


teams = input_data.home_team.dropna().drop_duplicates().values


# In[3]:


# teams = ['Yeclano','Bnei Yehuda']
score = Scoring(data = df, teams = teams )


# In[ ]:


elo_df = score.main()
