# harvard and mit article
# https://poseidon01.ssrn.com/delivery.php?ID=333026086024086016016095027067122093053092066027063087014082073064068075100104103006097025058123057012116084126112084094066066122015029086009007084008004118080092119090058007101011096090124127126015025070107124079107086096007007016119101027122026022105&EXT=pdf

# sklearn regression metrics
# http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

# edx predictor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import seaborn as sns
import sklearn.metrics as metrics

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import label_binarize
from matplotlib.colors import ListedColormap

def encodeOutputVariable(y):
    labelencoder_Y_Origin = LabelEncoder()
    y = labelencoder_Y_Origin.fit_transform(y.astype(str))
    return y

def encodeCategoricalData(X, index):
    # encode categorical data
    labelencoder_X_Origin = LabelEncoder()
    X[:, index] = labelencoder_X_Origin.fit_transform(X[:, index].astype(str))
    return X    

def manualEncodeLongStrings(X, column):
    index = 0
    test = 0
    keys = {}
    for row in X:
        print(row[4])
        print(row[column])
        key = row[column].replace(", ", "").replace(" ", "")
        if (keys.get(key) == None):
            keys[key] = index
            index += 1
        X[test][column] = keys.get(key)
        test += 1
    return X
    
def encodeHotEncoder(X, numberOfCategories):
    onehotencoder = OneHotEncoder(categorical_features = [numberOfCategories])
    X = onehotencoder.fit_transform(X.astype(str)).toarray()    
    X = X[:, 1:]
    return X

def minimumValues(train):
    return [0 if math.isnan(x) else x for x in train]

def minimumDelay(x):
    return 0 if np.isnan(x) or x < 0 else x

def outputPredictorResults(y_test, y_pred, title):
    # output results for Multiple Linear Regression
    print(title, "Analysis and Results")
    print("Explained Variance Score: ", metrics.explained_variance_score(y_test, y_pred))
    print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_pred))
    print("Mean Squared Logarithmic Error: ", metrics.mean_squared_log_error(y_test, y_pred))
    print("Median Absolute Error: ", metrics.median_absolute_error(y_test, y_pred))
    print("R2 Score: ", metrics.r2_score(y_test, y_pred))
    
    
def graphRegressorResults(X_test, y_test, mult_y_pred):
    numCases = []
    [numCases.append(x) for x in range(0, len(X_test))]
    plt.scatter(numCases, y_test, color="red")
    plt.scatter(numCases, mult_y_pred, color="blue")
    plt.title("Predicted vs. Actual Certificates Rewarded")
    plt.xlabel("Class Case")
    plt.ylabel("Number of Certificates")
    plt.show()

# developing the Multiple Linear Regression
def creatingMultipleLinearRegressionPredictor(X_train, y_train, X_test, y_test):
    # initialize the Multi Layer Perceptron Neural Network 
    regressor = LinearRegression()
    
    # fitting the Multi Layer Perceptron to the training set
    regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    mult_y_pred = regressor.predict(X_test)
    
    print("Coefficients: ", regressor.coef_)
    
    # output results
    outputPredictorResults(y_test, mult_y_pred, "Multiple Linear Regression")
    graphRegressorResults(X_test, y_test, mult_y_pred)

# importing the data
dataset = pd.read_csv("./data/appendix.csv")
X = dataset.iloc[:, 0:].values    
X = np.delete(X, [4, 7, 10, 12, 14], axis=1)
y = dataset.iloc[:, 10].values

# encode categorical data
X = encodeCategoricalData(X, 0)
X = encodeCategoricalData(X, 1)
X = encodeCategoricalData(X, 2)
X = encodeCategoricalData(X, 3)
X = encodeCategoricalData(X, 4)

X = encodeHotEncoder(X, 4)
y = encodeOutputVariable(y)

## splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

## feature scaling 
sc = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## outputting data summary
print("Summary Info About the Dataset")
print("Does category contain null values?")
print(dataset.isnull().any(), "\n")

creatingMultipleLinearRegressionPredictor(X_train, y_train, X_test, y_test)
