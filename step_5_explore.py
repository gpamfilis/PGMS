# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

# try:
#     result_key = sys.argv[1]
#     side_home_or_away = sys.argv[2]
# except Exception as e:
#     print(e)
#     print('using default')
result_key = 'over_under_0.5'
side_home_or_away = 'away_team'

# pd.read_csv()
dfh = pd.read_csv('./data/elo_ratings/results/regular/'+'home_team'+'/'+result_key+'/'+'metrics.csv', index_col='name')
dfa = pd.read_csv('./data/elo_ratings/results/regular/'+'away_team'+'/'+result_key+'/'+'metrics.csv', index_col='name')
dfh.head()
dfall = pd.concat([dfh,dfa],axis=1)
dfall.columns = ['hoac','hong','awac','awng']

dfall.columns




dfall.shape

df1 = dfall[(dfall.hong>10)&(dfall.awng>10)]
df1[['hoac','awac']].plot(kind='box')



dfh.head()
dfa.head()


df.accuracy.plot(kind='box')
plt.show()

#filter out
df1 = df[df['accuracy']>0.5]
df1.accuracy.plot(kind='box')
plt.show()


df2 = df[(df['accuracy']>0.5)&(df['n_games']>10)]
df2['accuracy'].plot(kind='box')

plt.show()

df.shape
df1.shape
df2.shape
