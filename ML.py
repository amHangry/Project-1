import pandas as pd
import csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

training_values=pd.read_csv("dengue_features_train_ndvi_avgmedain_no_start.csv")
y=pd.read_csv("dengue_labels_train.csv").total_cases
feature_columns=['city','reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_dew_point_temp_k',
                 'station_avg_temp_c',
                 'station_min_temp_c']
X=training_values[feature_columns]
X1=pd.read_csv("dengue_features_test.csv")[feature_columns]
model=RandomForestClassifier(n_estimators=2000)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model.fit(train_X, train_y)
# get predicted prices on validation data
val_predictions = model.predict(val_X)
#print(mean_absolute_error(val_y, val_predictions))
an=model.predict(X1)

with open('ans.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in an:
        writer.writerow([str(i)])
