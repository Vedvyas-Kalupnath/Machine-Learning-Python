import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "C:\\Users\\pravish\\PycharmProjects\\Linear\\iris.data"

headernames = ['sepal-length','sepal-width','petal-length','petal-width','class']

dataset = pd.read_csv(path,names = headernames)
dataset.head()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x , y, test_size = 0.40)

from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
Scaler.fit(X_train)
X_train = Scaler.transform(X_train)
X_test = Scaler.transform(X_test)

from sklearn.neighbors import  KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test,y_pred)
print("classification report:")
print(result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

