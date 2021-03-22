#This code is inspired by an article by Robert Clark. You can read his article at https://towardsdatascience.com/predict-college-basketball-scores-in-30-lines-of-python-148f6bd71894
from builddataset import build_train_test_split, inverse_scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas

X_train, X_test, y_train, y_test = build_train_test_split(["home_points","away_points","home_won"], ["home_won"])

parameters = {'activation': 'tanh',
              'hidden_layer_sizes': (80, 100, 80),
              'learning_rate': 'adaptive',
              'max_iter': 1400,
              'solver': 'sgd'}

model = MLPClassifier(**parameters)
model.fit(X_train, y_train[:, 0])

results = model.predict(X_test)

#Let's convert y_test into a one dimentional array so that it can be more easily used to validate the results!
y_test = y_test[:, 0]

df = pandas.DataFrame()
df["actual_winner"] = ["Home" if winner else "Away" for winner in y_test]
df["predicted_winner"] = ["Home" if winner else "Away" for winner in results]

with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

print(f"Accuracy when determining the winner: {round(sum(y_test == results)/len(results),4)*100}%")
