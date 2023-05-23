"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Valencia_temp', 'Seville_temp',
       'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max',
       'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp',
       'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min',
       'Madrid_temp', 'Madrid_temp_min', 'year', 'month', 'day', 'hour']]

# Fit model
rfr = RandomForestRegressor()
print ("Training Model...")
rfr.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/rfr_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(rfr, open(save_path,'wb'))
