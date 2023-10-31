# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated()
function respectively.
3.LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules
from sklearn.
7.Apply new unknown values.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Roghith K
RegisterNumber:212222040135
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![1.Placement Data](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/37479a83-ec5e-45be-887c-0566ea2ed85c)
![2.Salary Data](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/0fbedf2e-1bb7-4da1-a27b-8569fe8b3956)
![3.Checking the null() function](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/f44d5792-b310-4bad-8b79-c691141627de)
![4.Data Duplicate](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/231b91c2-4c5d-4c53-949c-e1d5424d3d13)
![5.Print Data](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/940c68f2-fe79-4e7c-a5fa-95927eb14d23)
![6.Data-status](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/46de19ef-d7d8-492b-967a-1cba8ebff481)

![7.y_prediction array](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/71d15e5f-b161-4e0f-a09c-597788f2603a)
![8.Accuracy Value](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/ea13a823-804d-43d9-807b-1bdcdc65dc8d)
![9.Confusion Array](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/a2cc8a21-49d0-4dad-b2a2-742af80e2dc5)
![10.Classification Report](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/4da43990-52a5-4b11-a999-48e50244083e)
![11.Prediction of LR](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/18948c86-c5dd-4164-b9d0-7eb544593f42)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
