# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
![273833218-c4603361-a589-4a4d-9454-adc780f5317a](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/37479a83-ec5e-45be-887c-0566ea2ed85c)
![273833240-6a05087a-896f-4dda-9183-117d0959c138](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/94b4d4da-77e1-497a-9c1a-ae374b529b77)
![273833281-f15df1d5-ad1b-4207-9fb1-1ef4bec98351](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/0fbedf2e-1bb7-4da1-a27b-8569fe8b3956)
![273833483-ec6f4dec-33eb-4213-9db7-826f3adc64c8](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/f44d5792-b310-4bad-8b79-c691141627de)
![273833507-85defd42-a247-4168-812c-e665dfd40f30](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/231b91c2-4c5d-4c53-949c-e1d5424d3d13)
![273833536-ce6b2dde-45b4-4095-9a07-e823550cb8df](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/940c68f2-fe79-4e7c-a5fa-95927eb14d23)
![273833563-f1eef88f-1fe9-4e55-92ba-d291dfe05bf4](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/46de19ef-d7d8-492b-967a-1cba8ebff481)

![273833588-4e9d873c-53db-4c48-aaf9-d17c864bcb46](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/71d15e5f-b161-4e0f-a09c-597788f2603a)
![273833606-edc84ac3-fa66-4b6b-a67c-e71f61e8e456](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/ea13a823-804d-43d9-807b-1bdcdc65dc8d)
![273833626-dd5ee355-0cde-41f6-a5c2-3d5e3a9b867e](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/a2cc8a21-49d0-4dad-b2a2-742af80e2dc5)
![273833652-1707abcb-487f-4c61-a0cf-738bc416509f](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/4da43990-52a5-4b11-a999-48e50244083e)
![273833726-bd376339-df10-489b-b05e-7005289d893e](https://github.com/RoghithKrishnamoorthy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475474/18948c86-c5dd-4164-b9d0-7eb544593f42)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
