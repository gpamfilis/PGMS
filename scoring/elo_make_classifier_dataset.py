from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pprint

import pandas as pd
import numpy as np

import h5py
import progressbar

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from scoring.utils import get_match_indexes
from pack.utils import delete_directory_contents



def get_elo_home_and_away(data_h5, row, ix):
    """
    This functions takes the data from the pair. iterates over the main data
    and finds the coresponding elo ratings based on the side the team is on.
    """
    hdhome = data_h5[row.home_team][:].T
    hdaway = data_h5[row.away_team][:].T
    elohome = hdhome[np.where(hdhome==ix)[1],1]
    eloaway = hdaway[np.where(hdaway==ix)[1],1]
    return elohome, eloaway


if __name__ == '__main__':

    print('[LOADING...]: Main DataFrame.')
    df = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv', index_col='Unnamed: 0')
    pred_data = pd.read_csv('../data/predict.csv', index_col='Unnamed: 0')
    df_teams = pred_data[['home_team','away_team']].values

    for p, pair in enumerate(df_teams[:]):

        print('[INFO]: pairs: ', pair)

        h5f = h5py.File('./elo_temp/elo_pairs_' + str(p) + '.h5', 'r')
        pair_num = list(h5f.keys())[0]
        # NOTE: do i need to convert this to a list since i iterate over it?
        teams = list(h5f[pair_num].keys())
        data = h5f[pair_num]
        print('[INFO]: getting indexes')
        uni = get_match_indexes(data_h5=data, teams=teams)

        da = df.loc[uni][['home_team','away_team']]
        da['res_index'] = uni
        bar = progressbar.ProgressBar(max_value=da.shape[0],widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])
        print('[INFO]: finding ratings.')
        elos = np.zeros((da.shape[0], 2))
        for i, (ix, row) in enumerate(da.iloc[:].iterrows()):
            # if i%100==0:
            #     print(i,da.shape[0])
            try:
                elohome, eloaway = get_elo_home_and_away(row=row, ix=ix, data_h5=data)
                elos[i,0] = elohome
                elos[i,1] = eloaway
            except Exception as e:
                elos[i,0] = np.nan
                elos[i,1] = np.nan
            bar.update(i)

        elos_df = pd.DataFrame(elos, columns=['EH','EA'], index=da.index)

        final = pd.concat([da,elos_df],axis=1)
        # NOTE: do not drop duplicates just in case i am able to impute.
        # final = final.dropna()

        final.to_csv('./elo_goals_data_for_classifier/data_'+str(p)+'.csv')
        h5f.close()
