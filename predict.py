#This code is inspired by an article by Robert Clark. You can read his article at https://towardsdatascience.com/predict-college-basketball-scores-in-30-lines-of-python-148f6bd71894
from builddataset import X_train, X_test, y_train, y_test, inverse_scale
from sklearn.ensemble import RandomForestRegressor
import pandas, numpy

#The random forest's hyperparameters
parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

#Now train the model...
model = RandomForestRegressor(**parameters)
model.fit(X_train, y_train)

#... and get the predictions!
results = model.predict(X_test)

#The values are currently scaled between 0 and 1. This scales them back to their original values
results = numpy.array([inverse_scale(results[:,0],'home_points'),inverse_scale(results[:,1],'away_points')]).T
y_test = numpy.array([inverse_scale(y_test[:,0],'home_points'),inverse_scale(y_test[:,1],'away_points')]).T

#Store the results in a DataFrame
df = pandas.DataFrame()
df["actual_winner"] = ["Home" if winner[0] > winner[1] else "Away" for winner in y_test]
df["predicted_winner"] = ["Home" if winner[0] > winner[1] else "Away" for winner in results]

#Print the entire DataFrame
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

#Iterate through the results to see which team the model predicted would win
predicted_winner = [winner[0] > winner[1] for winner in results]
actual_winner = [winner[0] > winner[1] for winner in y_test]
prediction_correct = [p == r for p,r in zip(predicted_winner, actual_winner)]

#Print the accuracy of determining the winner
print(f"Accuracy when determining the winner: {round(sum(prediction_correct)/len(results),4)*100}%")

#Now let's see how far off the model is from the actual score

#Please never use list comprehension like this
#               [1]                                          [2]                                         [3]                         [4]                    [5]                 
absolute_diff = [score for score_tuple in [(abs(home_p - home_r),abs(away_p - away_r)) for (home_p,away_p),(home_r,away_r) in zip(results,y_test)] for score in score_tuple]
#[1] return the reasult
#[2] calculate the difference between the (p)rediction and the (r)eal value
#[3] doubly-nested unpacking of the predicted and actual game scores
#[4] zip the predicted and actual game scores together so I can unpack them easier
#[5] Flatten the list because I'm returning a tuple.

print(f"Average absolute deviation from the real score: {round(sum(absolute_diff)/len(absolute_diff),2)}")
