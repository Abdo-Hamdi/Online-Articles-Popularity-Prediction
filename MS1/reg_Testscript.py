import numpy as np
import pandas as pd
from pickle import dump, load
from scipy import stats
from sklearn import linear_model
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from tkinter import filedialog
from sklearn.metrics import r2_score

# import TestScript

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
    '[]': 6.0,
    'unknown_1': 7.0,
    'unknown_2': 8.0
}
f = open('channel_type_mapping.pkl', 'wb')
dump(channel_type_mapping, f)
f.close()

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
f1 = open('weekday_mapping.pkl', 'wb')
dump(weekday_mapping, f1)
f1.close()

isWeekEnd = {
    'No': 0.0,
    'Yes': 1.0,
    'unknown_1': 2.0,
    'unknown_2': 3.0
}
f2 = open('isWeekEnd.pkl', 'wb')
dump(isWeekEnd, f2)
f2.close()


columns_to_scale = [col for col in DF.columns if col not in ['channel type', 'weekday', 'isWeekEnd']]
scaler = MinMaxScaler()
DF[columns_to_scale] = scaler.fit_transform(DF[columns_to_scale])
f3 = open('scaler.pkl', 'wb')
dump(scaler, f3)
f3.close()

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

f5 = open('linearRegression.pkl', 'wb')
dump(sln, f5)
f5.close()

poly_features = PolynomialFeatures(degree=2)
f7 = open('polynomialFeatures.pkl', 'wb')

X_train_poly = poly_features.fit_transform(X_train)

dump(poly_features, f7)
f7.close()

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)
y_train_predicted = poly_model.predict(X_train_poly)
y_test_predicted = poly_model.predict(poly_features.transform(X_test))

print('--- Polynomial Regression ---')
# print('Co-efficient of linear regression', poly_model.coef_)
# print('Intercept of linear regression model', poly_model.intercept_)
print('Mean Square Error Train Model 2 : ', mean_squared_error(Y_train, y_train_predicted))
print('Mean Square Error Test Model 2 : ', mean_squared_error(Y_test, y_test_predicted))

f6 = open('polynomialRegression.pkl', 'wb')
dump(poly_model, f6)
f6.close()


def testScriptReg(Data):
    Data = Data.drop(columns=['title', 'url', ' timedelta'])

    with open('channel_type_mapping.pkl', 'rb') as f:
        channelType = load(f)
    with open('weekday_mapping.pkl', 'rb') as f:
        weekDay = load(f)
    with open('isWeekEnd.pkl', 'rb') as f:
        isWeekEnd = load(f)
    with open('linearRegression.pkl', 'rb') as f:
        linearReg = load(f)
    with open('polynomialRegression.pkl', 'rb') as f:
        polyReg = load(f)
    with open('polynomialFeatures.pkl', 'rb') as f:
        polyFeatures = load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = load(f)

    Data['weekday'] = Data['weekday'].map(weekDay)
    Data['channel type'] = Data['channel type'].map(channelType)
    Data['isWeekEnd'] = Data['isWeekEnd'].map(isWeekEnd)

    columns_to_scale = [col for col in Data.columns if col not in ['channel type', 'weekday', 'isWeekEnd']]
    Data[columns_to_scale] = scaler.transform(Data[columns_to_scale])

    def detect_outliers_z_score(data):
        columns_to_check = [col for col in data.columns if col not in ['channel type', 'weekday', 'isWeekEnd']]
        outliers = {}
        threshold = 3
        for col in data.columns:
            if col in columns_to_check:
                z_scores = np.abs(stats.zscore(data[col]))
                outliers[col] = data.index[z_scores > threshold].tolist()
        return outliers

    outliers = detect_outliers_z_score(Data)

    for col, indices in outliers.items():
        for index in indices:
            Data.at[index, col] = Data[col].mean()

    null_columns_train = Data.columns[Data.isnull().any()]
    for col in null_columns_train:
        Data[col].fillna(Data[col].mean(), inplace=True)

    X_test = Data.iloc[:, :-1]
    Y_test = Data.iloc[:, -1]

    prediction = linearReg.predict(X_test)
    print('Mean Square Error Test Model 1 : ', mean_squared_error(Y_test, prediction))
    print('R2 Score Test Model 2 : ', r2_score(Y_test, prediction))

    X_test_poly = polyFeatures.transform(X_test)
    y_test_predicted = polyReg.predict(X_test_poly)
    print('Mean Square Error Test Model 2 : ', mean_squared_error(Y_test, y_test_predicted))
    print('R2 Score Test Model 1 : ', r2_score(Y_test, y_test_predicted))


while True:
    x = input("Do You Want To Use Test Script?!(y/n): ")
    if x == 'y' or x == 'Y':
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.csv")])
        data = pd.read_csv(file_path)
        print(file_path)
        testScriptReg(data)
        break
    else:
        break
