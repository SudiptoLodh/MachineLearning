#import numpy as np
#import sklearn
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd
import PlotLearningCurve

import seaborn as sns

def Create_Breast_Cancer_Winconsin_Dataset():
    ########## Breast Cancer DataSet Starts ###################
    # Data Set Location
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

    # load the dataset
    data = pd.read_csv("breast-cancer-wisconsin.data", names=["ID",
                                                              "Clump Thickness",
                                                              "Uniformity of Cell Size",
                                                              "Uniformity of Cell Shape",
                                                              "Marginal Adhesion",
                                                              "Single Epithelial Cell Size",
                                                              "Bare Nuclei",
                                                              "Bland Chromatin",
                                                              "Normal Nucleoli",
                                                              "Mitoses",
                                                              "Class"])
    # Preprocess the data
    data.replace('?', -99999, inplace=True)
    # Split the Training the Target Data from the Dataset
    columns = data.columns.tolist()
    columns = [c for c in columns if c not in ["Class", "ID"]]
    target = "Class"
    ####### Breast Cancer DataSet ##############
    return data,target,columns

def Create_Wine_Dataset():
    ########## Wine Dataset Starts ################
    data = pd.read_csv("wine.data", names=["ID",
                                           "Alcohol",
                                           "Malic_acid",
                                           "Ash",
                                           "Alcalinity_of_ash",
                                           "Magnesium",
                                           "Total_phenols",
                                           "Flavanoids",
                                           "Nonflavanoid_phenols",
                                           "Proanthocyanins",
                                           "Color intensity",
                                           "Hue",
                                           "OD280_OD315_diluted_wines",
                                           "Proline"])
    columns = data.columns.tolist()
    columns = [c for c in columns if c not in ["ID"]]
    target = "ID"
    ################ Wine Dataset Ends ##############
    return data,target,columns

def Create_Adult_Dataset():
    ################ Adult Data ######################
    data = pd.read_csv("adult.data", names=["age",
                                            "workclass",
                                            "fnlwgt",
                                            "education",
                                            "education-num",
                                            "marital-status",
                                            "occupation",
                                            "relationship",
                                            "race",
                                            "sex",
                                            "capital-gain",
                                            "capital-loss",
                                            "hours-per-week",
                                            "native-country",
                                            "Salary"])

    LE = preprocessing.LabelEncoder()
    data = data.apply(LE.fit_transform)
    columns = data.columns.tolist()
    columns = [c for c in columns if c not in ["Salary"]]
    target = "Salary"
    data.replace('?', None, inplace=True)
    ############# Adult Dataset ends #################
    return data,target,columns,LE

data,target,columns = Create_Breast_Cancer_Winconsin_Dataset()
X=data[columns]
y=data[target]

#Split the data in Training set and Testing set with 0.2 ratio
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)
#Specify the testing parameters
seed = 5
scoring = "accuracy"
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

#Start Training the Set
models = []
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC(gamma='auto')))
models.append(('Decision_Tree',DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=10, min_samples_leaf=5,ccp_alpha=0.005)))
models.append(('GradientBoost',GradientBoostingClassifier()))
models.append(('Neural_Network',MLPClassifier(alpha=1, max_iter=1000)))

results = []
names = []

for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
    cv_results = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print("CancerDataSet "+msg)

    fig, axes = plt.subplots(2,1, figsize=(10, 15))
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    PlotLearningCurve.plot_learning_curve(model, name, X, y, axes=axes, ylim=(0.1, 1.01),
                                          cv=cv, n_jobs=4)
    plt.savefig("CancerDataSet"+name);

data, target, columns = Create_Wine_Dataset()
X = data[columns]
y = data[target]

# Split the data in Training set and Testing set with 0.2 ratio
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# Specify the testing parameters
seed = 5
scoring = "accuracy"

# Start Training the Set
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC(gamma='auto')))
models.append(('Decision_Tree', DecisionTreeClassifier(criterion="gini",
                                                           random_state=100, max_depth=10, min_samples_leaf=5,
                                                           ccp_alpha=0.005)))
models.append(('GradientBoost', GradientBoostingClassifier()))
models.append(('Neural_Network', MLPClassifier(alpha=1, max_iter=1000)))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print("Wine " + msg)

    fig, axes = plt.subplots(2, 1, figsize=(10, 15))
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    PlotLearningCurve.plot_learning_curve(model, name, X, y, axes=axes, ylim=(0.1, 1.01),
                                              cv=cv, n_jobs=4)
    plt.savefig("Wine " + name);


#Testing with Custom Dataset
data, target, columns,LE = Create_Adult_Dataset()

X = data[columns]
y = data[target]
model = DecisionTreeClassifier(criterion="gini",
                               random_state=100, max_depth=10, min_samples_leaf=5,ccp_alpha=0.005)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
model.fit(X_train,y_train)
plot_confusion_matrix(model,X_test,y_test)
plt.show()
plt.savefig("Adult_Dataset_ConfusionMatrix")
y_predicted = model.predict(X_test)
le = preprocessing.LabelEncoder()
print(classification_report(y_true=y_test, y_pred=y_predicted))



