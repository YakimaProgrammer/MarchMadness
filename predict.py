#This code is inspired by an article by Robert Clark. You can read his article at https://towardsdatascience.com/predict-college-basketball-scores-in-30-lines-of-python-148f6bd71894
from builddataset import build_train_test_split, inverse_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = build_train_test_split(["home_points","away_points","home_won"], ["home_won"])

parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

model = RandomForestClassifier(**parameters)
model.fit(X_train, y_train)
results = model.predict(X_test)

#Let's convert y_test into a one dimentional array so that it can be more easily used to validate the results!
y_test = y_test[:, 0]

print("Predicted winner:\tActual:")
for prediction,real in zip(results, y_test):
    print("\t\t","Home" if prediction else "Away","\t","Home" if real else "Away")
    
print(f"Accuracy when determining the winner: {round(sum(results == y_test)/len(results),4)*100}%")
