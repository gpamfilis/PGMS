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
dfs = []
for i in range(50):
    df = pd.read_csv('./data/yo_'+str(i)+'.csv', index_col=0)
    dfs.append(df)

all_teams = pd.concat(dfs, axis=0)
all_teams.columns = ['name','ac','ng']
# all_teams.head(2)


all_teams = all_teams.sort_values(by=['ac'], ascending=False)
initial_shape = all_teams.shape[0]

print(initial_shape)

f1 = all_teams[(all_teams.ng>60) & (all_teams.ac>0.6)]
print(f1.shape[0])

scipy.stats.pearsonr(f1.ng.values, f1.ac.values)

plt.scatter(f1.ng.values, f1.ac.values)
plt.show()





data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
teh = data.groupby('home_team').size().to_frame()
d = teh[(teh>10) & (teh<300)].dropna()
pd.DataFrame(d[0].values).plot(kind='hist')
plt.show()

d[0].plot(kind='box', showmeans=True)
plt.show()
