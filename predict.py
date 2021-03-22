#This code is inspired by an article by Robert Clark. You can read his article at https://towardsdatascience.com/predict-college-basketball-scores-in-30-lines-of-python-148f6bd71894
from builddataset import build_train_test_split, inverse_scale
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#Only for this test
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas

X_train, X_test, y_train, y_test = build_train_test_split(["home_points","away_points","home_won"], ["home_won"])

parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', ',adam'],
              'hidden_layer_sizes': [(50, 70, 50), (60, 80, 60), (70, 90, 70), (80, 100, 80), (90, 110, 90), (100, 120, 100), (110, 130, 110), (120, 140, 120), (130, 150, 130), (70, 50), (80, 60), (90, 70), (100, 80), (110, 90), (120, 100), (130, 110), (140, 120), (150, 130), (70,), (80,), (90,), (100,), (110,), (120,), (130,), (140,), (150,)],
              'max_iter': list(range(100, 1500, 50)),
              'alpha': [0.0001, 0.005, 0.01, 0.05],
              'learning_rate': ['constant','adaptive'],
              }

model = MLPClassifier()
clf = RandomizedSearchCV(model, parameters, n_jobs=-1, verbose = 3, n_iter = 5)
clf.fit(X_train, y_train[:, 0])
