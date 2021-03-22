#This code is inspired by an article by Robert Clark. You can read his article at https://towardsdatascience.com/predict-college-basketball-scores-in-30-lines-of-python-148f6bd71894
from builddataset import build_train_test_split, inverse_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas

X_train, X_test, y_train, y_test = build_train_test_split(["home_points","away_points","home_won"], ["home_won"])

parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

model = RandomForestClassifier(**parameters)
model.fit(X_train, y_train[:, 0])

results = model.predict_proba(X_test)

#Let's convert y_test into a one dimentional array so that it can be more easily used to validate the results!
y_test = y_test[:, 0]

df = pandas.DataFrame()
df["actual_winner"] = ["Home" if winner else "Away" for winner in y_test]
df["predicted_winner"] = ["Home" if winner[0] > winner[1] else "Away" for winner in results]
df["home_winning_chance"] = results[:, 0]
df["away_winning_chance"] = results[:, 1]

with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)


predicted_winner = (results[:, 0] > results[:, 1]).astype(int)
print(f"Accuracy when determining the winner: {round(sum(y_test == predicted_winner)/len(predicted_winner),4)*100}%")
