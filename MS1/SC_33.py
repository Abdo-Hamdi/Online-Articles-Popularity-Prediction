import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

DF = pd.read_csv('OnlineArticlesPopularity.csv')
# DF.info()
# print(DF.duplicated())
DF = DF.drop(columns=['title', 'url', ' timedelta'])
channel_type_mapping = {
    ' data_channel_is_bus': 0.0,
    ' data_channel_is_socmed': 1.0,
    ' data_channel_is_lifestyle': 2.0,
    ' data_channel_is_world': 3.0,
    ' data_channel_is_entertainment': 4.0,
    ' data_channel_is_tech': 5.0,
    '[]': 6.0
}
weekday_mapping = {
    'monday': 0.0,
    'tuesday': 1.0,
    'wednesday': 2.0,
    'thursday': 3.0,
    'friday': 4.0,
    'saturday': 5.0,
    'sunday': 6.0
}
isWeekEnd = {
    'No': 0.0,
    'Yes': 1.0
}

columns_to_scale = [col for col in DF.columns if col not in ['channel type', 'weekday', 'isWeekEnd', 'shares']]

scaler = MinMaxScaler()

DF[columns_to_scale] = scaler.fit_transform(DF[columns_to_scale])

def detect_outliers_z_score(data):
    columns_to_check = [col for col in DF.columns if col not in ['channel type', 'weekday', 'isWeekEnd']]
    outliers = {}
    threshold = 3
    for col in data.columns:
        if col in columns_to_check:
            z_scores = np.abs(stats.zscore(data[col]))
            outliers[col] = data.index[z_scores > threshold].tolist()
    return outliers


outliers = detect_outliers_z_score(DF)
for col, indices in outliers.items():
    for index in indices:
        DF.at[index, col] = DF[col].mean()

DF['weekday'] = DF['weekday'].map(weekday_mapping)
DF['channel type'] = DF['channel type'].map(channel_type_mapping)
DF['isWeekEnd'] = DF['isWeekEnd'].map(isWeekEnd)

corr = DF.corr()
top_feature = corr.index[abs(corr[' shares']) > 0.1]
top_corr = DF[top_feature].corr()

# sns.heatmap(top_corr, annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
# plt.show()

X = DF.iloc[:, :-1]
Y = DF.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=10, shuffle=True, random_state=15)

# Linear Regression Model
sln = linear_model.LinearRegression()
sln.fit(X_train, Y_train)
y_predicted = sln.predict(X_train)
prediction = sln.predict(X_test)

print('--- Linear Regression ---')
# print('Co-efficient : ', sln.coef_)
# print('Intercept : ', sln.intercept_)
print('Mean Square Error Train Model 1 : ', mean_squared_error(Y_train, y_predicted))
print('Mean Square Error Test Model 1 : ', mean_squared_error(Y_test, prediction))
print('R2 Score Train Model 1 : ', r2_score(Y_train, y_predicted) * 100)
print('R2 Score Test Model 1 : ', r2_score(Y_test, prediction) * 100)

# Polynomial Regression Model
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)
y_train_predicted = poly_model.predict(X_train_poly)
y_test_predicted = poly_model.predict(poly_features.transform(X_test))

print('--- Polynomial Regression ---')
# print('Co-efficient of linear regression', poly_model.coef_)
# print('Intercept of linear regression model', poly_model.intercept_)
print('Mean Square Error Train Model 2 : ', mean_squared_error(Y_train, y_train_predicted))
print('Mean Square Error Test Model 2 : ', mean_squared_error(Y_test, y_test_predicted))
print('R2 Score Train Model 2 : ', r2_score(Y_train, y_train_predicted) * 100)
print('R2 Score Test Model 2 : ', r2_score(Y_test, y_test_predicted) * 100)
