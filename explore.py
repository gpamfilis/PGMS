import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import seaborn as sns
import numpy as np


print('Reading Champs...')
champs = pd.read_csv('./champs.csv')

correlations = []
index = []
for i in range(10):
    df = pd.read_csv('./data/elo_ratings/win_lose/yo_'+str(i)+'.csv', index_col=0)
    # df = pd.read_csv('./data/elo_ratings/goal_difference/yo_'+str(i)+'.csv', index_col=0)
    inx = champs.name[i]
    df.columns = ['name', 'ac', 'ng']
    df = df.sort_values(by=['ng'], ascending=False)
    df2 = df[(df.ac > 0.5) & (df.ng > 60)]
    if df2.shape[0] < 20:
        continue
    cor = scipy.stats.pearsonr(df2.ng.values, df2.ac.values)
    correlations.append(cor[0])
    index.append(inx)

df = pd.DataFrame(correlations, columns=['co'], index=index)
df = df.sort_values(by=['co'])
df.plot(kind='bar')
plt.show()
