#This code is inspired by an article by Robert Clark. You can read his article at https://towardsdatascience.com/predict-college-basketball-scores-in-30-lines-of-python-148f6bd71894
from builddataset import build_train_test_split, inverse_scale
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas, numpy

X_train, X_test, y_train, y_test = build_train_test_split(["home_points","away_points","home_won"], ["home_points","away_points"])

parameters = {'activation': 'tanh',
              'hidden_layer_sizes': (80, 100, 80),
              'learning_rate': 'adaptive',
              'max_iter': 1400,
              'solver': 'sgd'}

model = MLPRegressor(**parameters)
model.fit(X_train, y_train)

results = model.predict(X_test)

results = numpy.array([inverse_scale(results[:,0],'home_points'),inverse_scale(results[:,1],'away_points')]).T
y_test = numpy.array([inverse_scale(y_test[:,0],'home_points'),inverse_scale(y_test[:,1],'away_points')]).T

df = pandas.DataFrame()
df["actual_winner"] = ["Home" if winner[0] > winner[1] else "Away" for winner in y_test]
df["predicted_winner"] = ["Home" if winner[0] > winner[1] else "Away" for winner in results]

with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

predicted_winner = [winner[0] > winner[1] for winner in results]
actual_winner = [winner[0] > winner[1] for winner in y_test]
prediction_correct = [p == r for p,r in zip(predicted_winner, actual_winner)]

print(f"Accuracy when determining the winner: {round(sum(prediction_correct)/len(results),4)*100}%")
