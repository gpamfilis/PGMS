import progressbar
import pandas as pd
import os
import sys


dfs = []
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])

try:
    print('Received Chunk Params')
    n_start=int(sys.argv[1])
    n_end=int(sys.argv[2])
except Exception as e:
    print('[Running For All... good luck]')
    n_start=0
    n_end = len(os.listdir('./elo_goals_data_for_classifier'))


for p in bar(range(n_start, n_end)):
    df = pd.read_csv('./elo_goals_data_for_classifier/data_'+str(p)+'.csv', index_col='Unnamed: 0')
    dfs.append(df)
print('[INFO]: CONCATENATING')
final = pd.concat(dfs, ignore_index=True)
del dfs
print('[INFO]: droping nans')
final = final.dropna()
print('[INFO]: SAVING')
final.to_csv('./final.csv', index=True)
