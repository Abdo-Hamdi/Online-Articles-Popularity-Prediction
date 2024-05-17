import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LogisticRegression

DF = pd.read_csv("OnlineArticlesPopularity_Milestone2.csv")
DF = DF.drop(columns=['title', 'url'])

X = DF.iloc[:, :-1]
y = DF.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

channel_type_mapping = {
    ' data_channel_is_bus': 0.0,
    ' data_channel_is_socmed': 1.0,
    ' data_channel_is_lifestyle': 2.0,
    ' data_channel_is_world': 3.0,
    ' data_channel_is_entertainment': 4.0,
    ' data_channel_is_tech': 5.0,
    '[]': 6.0,
    'unknown_1': 7.0,
    'unknown_2': 8.0
}
weekday_mapping = {
    'monday': 0.0,
    'tuesday': 1.0,
    'wednesday': 2.0,
    'thursday': 3.0,
    'friday': 4.0,
    'saturday': 5.0,
    'sunday': 6.0,
    'unknown_1': 7.0,
    'unknown_2': 8.0
}
isWeekEnd = {
    'No': 0.0,
    'Yes': 1.0,
    'unknown_1': 2.0,
    'unknown_2': 3.0
}

# Replace strings in the columns with integers using the mapping
X_train['weekday'] = X_train['weekday'].map(weekday_mapping)
X_train['channel type'] = X_train['channel type'].map(channel_type_mapping)
X_train['isWeekEnd'] = X_train['isWeekEnd'].map(isWeekEnd)

# Preprocessing for testing data
X_test['weekday'] = X_test['weekday'].map(weekday_mapping)
X_test['channel type'] = X_test['channel type'].map(channel_type_mapping)
X_test['isWeekEnd'] = X_test['isWeekEnd'].map(isWeekEnd)

# Perform a Standard Scale on the selected columns
scaler = StandardScaler()
columns_to_scale = [col for col in X_train.columns if col not in [' Above Average']]
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# detect the outliers in each column using z-score
def detect_outliers_z_score(data):
    outliers = {}
    threshold = 3
    for col in data.columns:
        if col in columns_to_scale:
            z_scores = np.abs(stats.zscore(data[col]))
            outliers[col] = data.index[z_scores > threshold].tolist()
    return outliers

# Detect outliers in training data
outliers_train = detect_outliers_z_score(X_train)
# Change the outliers with the mean of the specified column
for col, indices in outliers_train.items():
    for index in indices:
        X_train.at[index, col] = X_train[col].mean()

# Detect outliers in testing data
outliers_test = detect_outliers_z_score(X_test)
# Change the outliers with the mean of the specified column
for col, indices in outliers_test.items():
    for index in indices:
        X_test.at[index, col] = X_test[col].mean()


null_columns_train = X_train.columns[X_train.isnull().any()]
null_columns_test = X_test.columns[X_test.isnull().any()]
for col in null_columns_train:
    X_train[col].fillna(X_train[col].mean(), inplace=True)
for col in null_columns_test:
    X_test[col].fillna(X_test[col].mean(), inplace=True)


C = 1.5
logistic_regression_model = LogisticRegression(C=C, max_iter=1000).fit(X_train, y_train)
Y_prediction_Train = logistic_regression_model.predict(X_train)
acc_logistic_train = accuracy_score(y_train, Y_prediction_Train)
print("Train Accuracy (Logistic Regression):", acc_logistic_train)

Y_prediction_Test = logistic_regression_model.predict(X_test)
acc_logistic_test = accuracy_score(y_test, Y_prediction_Test)
print("Test Accuracy (Logistic Regression):", acc_logistic_test)
