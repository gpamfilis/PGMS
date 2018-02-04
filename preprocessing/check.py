import pandas as pd

df = pd.read_csv('./data/elo_ratings/ratings/Albania - Cup.csv', index_col='Unnamed: 0')

team = df[df.columns[0]].dropna()
data = pd.read_csv('/home/kasper/Dropbox/Scrapping/soccerway/csv/final_data_soccerway.csv')
data.loc[120817]
