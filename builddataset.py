#This file downloads, parses, cleans, and saves a dataset of over 500 basketball games for further analysis
import pandas as pd
import numpy as np
from sportsreference.nba.teams import Teams
from sklearn.model_selection import train_test_split

FIELDS_TO_DROP = ['away_points', 'home_points', 'date', 'location',
                  'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
                  'winning_name']#, 'home_ranking', 'away_ranking']

try:
    with open("rawdataset.csv") as f:
        rawdataset = pd.read_csv(f)
        
except FileNotFoundError:
    rawdataset = pd.DataFrame()
    teams = Teams()
    for team in teams:
        try:
            rawdataset = pd.concat([rawdataset, team.schedule.dataframe_extended])
            print("[SUCCESS]", team)
        except IndexError:
            print("[FAILED]", team)
            
    rawdataset.to_csv("rawdataset.csv")

##Ok, something is weird with this dataset. "away_points" is always ~2x larger than "home_score"
##So, let's fix that!
rawdataset["away_points"] -= rawdataset["home_points"]

dataset = (rawdataset
    #remove rows with null values
    .dropna()
    #remove duplicate rows
    .drop_duplicates()
    #these features are irrelevant
    .drop(['Unnamed: 0', 'date', 'location', 'losing_abbr', 'losing_name', 'winner', 'winning_abbr', 'winning_name'], 1)
    #scale every feature between 0 and 1, independently of each other
    #if I scaled all of these, together between 0 and 1, I won't have done anything useful,
    #I'm trying to reduce bias towards features that tend to have larger numbers
    .apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
    #some of the models have trouble with float64 numbers 
    .astype(np.float32)
    #Some rows will now only contain NaN/Inf because of the previous scaling and resizing operations, so I need to replace all those values with 0
    .fillna(0)
)

X = dataset.drop(['away_points', 'home_points'], 1)
y = dataset[['home_points', 'away_points']].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
