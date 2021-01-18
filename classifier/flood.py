# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import preprocessing
import sklearn.metrics as metrics

dataset = pd.read_csv('data.csv')
df = dataset.drop(['Other Animals'], axis=1)



#scale = preprocessing.StandardScaler()
scale = preprocessing.Normalizer()
scale_df = scale.fit_transform(df)
#print(scale_df)
final_df = pd.DataFrame(scale_df)
#print(final_df)

X = final_df.iloc[:, :1]

y = final_df.iloc[:, -1]



#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model_flood_sc.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_flood_sc.pkl','rb'))
#print(model.predict([[2050]]))
#print(model.summary())
y_pred = regressor.predict(X)


#df = pd.DataFrame({'Actual': y, 'Predicted': y_pred.flatten()})

print(pd.DataFrame({'Actual': y, 'Predicted': y_pred.flatten()}))
print("########")
print(model.score(X,y))


print("########")
# Regression metrics
explained_variance=metrics.explained_variance_score(y, y_pred)
mean_absolute_error=metrics.mean_absolute_error(y, y_pred)
mse=metrics.mean_squared_error(y, y_pred)
mean_squared_log_error=metrics.mean_squared_log_error(y, y_pred)
median_absolute_error=metrics.median_absolute_error(y, y_pred)
r2=metrics.r2_score(y, y_pred)

print('explained_variance: ', round(explained_variance,4))
print('mean_squared_log_error: ', round(mean_squared_log_error,4))
print('r2: ', round(r2,4))
print('MAE: ', round(mean_absolute_error,4))
print('MSE: ', round(mse,4))
print('RMSE: ', round(np.sqrt(mse),4))
