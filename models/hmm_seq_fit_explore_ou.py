import pandas as pd
import numpy as np
# from os import sys, path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# from pack.utils import string_list_to_numpy
import matplotlib.pyplot as plt
%matplotlib inline


pd.read_csv('./data/elo_ratings/ratings/elo.csv')

all_teams = np.array(['a','b','c','d'])
all_teams
data = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]])
data

teams = np.array(['a','d'])
teams

ix = np.isin(all_teams, teams)
teams_ix = np.where(ix)[0]
final = data[:,teams_ix]



|def string_list_to_numpy(x):
    try:
        x = x.strip('[] ').split()
        ar = np.asarray(x).astype(float)
        if ar.shape[0]==1:
            return np.nan
        else:
            return ar
    except Exception as e:
        return np.nan

uo = ['over_under_0.5','over_under_1.5','over_under_2.5','over_under_3.5','over_under_4.5']

dfhas = []
for s in uo:
    print(s)
    dfh = pd.read_csv('./data/elo_ratings/results/regular/home_team/'+s+'/metrics.csv', index_col='name')
    dfa = pd.read_csv('./data/elo_ratings/results/regular/away_team/'+s+'/metrics.csv', index_col='name')
    dfha = pd.concat([dfh, dfa],axis=1).dropna()
    dfha.columns = ['hoac','hpx','hpxx','hong','awac','apx','apxx','awng']
    for c in ['hpx','apx']:
        dfha.loc[:,c] = dfha[c].apply(string_list_to_numpy)
    dfha = dfha[(dfha.hong>20) & (dfha.hoac>0.5) & (dfha.awac>0.5)]
    dfhas.append(dfha)

t = []
for i in range(len(uo)):
    t.append(dfhas[i]['hoac'])


df = pd.concat(t,axis=1)
df.columns = uo

df.plot(kind='box')
t = []
for i in range(len(uo)):
    t.append(dfhas[i]['awac'])


df = pd.concat(t,axis=1)
df.columns = uo

df.plot(kind='box')

team = dfhas[0].index[2]
for i in range(len(dfhas)):
    print(i)
    if team not in dfhas[i].index:
        print('Not!')



hpxs = []
for i in range(len(dfhas)):
    px = dfhas[i].loc[dfhas[0].index[0]]['hpx']
    hpxs.append(px)

apxs = []
for i in range(len(dfhas)):
    px = dfhas[i].loc[dfhas[0].index[2]]['apx']
    apxs.append(px)

home = np.array(hpxs)
away = np.array(apxs)

# pd.DataFrame(home,index=[0.5,1.5,2.5,3.5,4.5],columns=['o','u'])

1/home
1/away
(1/home).sum(axis=0)/(1/home).sum()
home
np.argmax(home,axis=1)
np.argmax(away,axis=1)


for i in range(len(pxs)-1):
    print(pxs[i][0]*pxs[i+1][1])


pxs[0][0]*pxs[0][0]*pxs[0][0]*pxs[0][1]*pxs[0][1]

pxs

pxs[0][0]*pxs[0][1]



1/pxs[0]
