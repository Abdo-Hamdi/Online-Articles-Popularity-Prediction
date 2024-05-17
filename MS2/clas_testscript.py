import numpy as np
import pandas as pd
from tkinter import filedialog
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib

def testScriptSVM(Data):
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
        'sunday': 6.0
    }
    isWeekEnd = {
        'No': 0.0,
        'Yes': 1.0,
        'unknown_1': 2.0,
        'unknown_2': 3.0
    }

    Data = Data.drop(columns=['title', 'url'])

    Data['weekday'] = Data['weekday'].map(weekday_mapping)
    Data['channel type'] = Data['channel type'].map(channel_type_mapping)
    Data['isWeekEnd'] = Data['isWeekEnd'].map(isWeekEnd)

    # Perform a Standard Scale on the selected columns
    scaler = StandardScaler()
    columns_to_scale = [col for col in Data.columns if col not in ['Article Popularity']]
    Data[columns_to_scale] = scaler.fit_transform(Data[columns_to_scale])

    # detect and handle outliers
    def detect_outliers_z_score(data):
        outliers = {}
        threshold = 3
        for col in data.columns:
            if col in columns_to_scale:
                z_scores = np.abs(stats.zscore(data[col]))
                outliers[col] = data.index[z_scores > threshold].tolist()
        return outliers

    outliers = detect_outliers_z_score(Data)
    # Change the outliers with the mean of the specified column
    for col, indices in outliers.items():
        for index in indices:
            Data.at[index, col] = Data[col].mean()

    # Replace null values with the mean
    null_columns = Data.columns[Data.isnull().any()]
    for col in null_columns:
        Data[col].fillna(Data[col].mean(), inplace=True)

    # Load SVM model
    svm_model = joblib.load('svm_model.pkl')

    X_test = Data.iloc[:, :-1]
    y_test = Data.iloc[:, -1]

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)

    while True:
        choice = input("Do you want to use the test script again? (y/n) : ")
        if choice.lower() == 'y':
            file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            new_data = pd.read_csv(file_path)
            testScriptSVM(new_data)
            break
        elif choice.lower() == 'n':
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")


test_data = pd.read_csv("OnlineArticlesPopularity_Milestone2.csv")
testScriptSVM(test_data)
