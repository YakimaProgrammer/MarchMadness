from builddataset import X_train, X_test, y_train, y_test, inverse_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}
model = RandomForestRegressor(**parameters)
model.fit(X_train, y_train)
results = model.predict(X_test)

results = zip(inverse_scale(results[:,0],'home_points'),inverse_scale(results[:,1],'away_points'))
y_test = zip(inverse_scale(y_test[:,0],'home_points'),inverse_scale(y_test[:,1],'away_points'))

print("Predicted: Home\tAway\tActual: Home\tAway")
winloss = []
for prediction,real in zip(results, y_test):
    winloss.append([[prediction[0] > prediction[1]],[real[0] > real[1]]])
    print("\t", round(prediction[0],3), "   \t", round(prediction[1],3), "  \t\t", round(real[0],3), "  \t", round(real[1],3))
    
print(f"Accuracy when determining the winner: {round(sum(p==r for p,r in winloss)/len(winloss),4)*100}%")
