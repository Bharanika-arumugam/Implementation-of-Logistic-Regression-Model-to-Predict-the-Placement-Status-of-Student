# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.  Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusio
4. Display the results.
```

## Program:
```
Developed by: BHARANIKA A.S
RegisterNumber:  212224040048
```
```
import pandas as pd
 from sklearn.preprocessing import LabelEncoder
 data = pd.read_csv('/content/Placement_Data.csv')
 data.head()
 data1 = data.copy()
 data1 = data1.drop(["sl_no", "salary"], axis = 1)
 data1.head()
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
 x = data1.iloc[:, :-1]
 x
 y = data1["status"]
 y
 from sklearn.model_selection import train_test_split
 
 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
from sklearn.linear_model import LogisticRegression
 lr = LogisticRegression(solver = "liblinear")
 lr.fit(x_train, y_train)
 y_pred = lr.predict(x_test)
 y_pred
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 accuracy
 from sklearn.metrics import confusion_matrix
 confusion = (y_test, y_pred)
 confusion
 from sklearn.metrics import classification_report
 classification_report1 = classification_report(y_test, y_pred)
 print(classification_report1)
 lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

## Placement Data:
![image](https://github.com/user-attachments/assets/0b16cc59-c3bd-4c15-adff-6a902d1b44c2)
## Checking null function:
![image](https://github.com/user-attachments/assets/e1acce53-2401-4890-a089-918cd1972d1f)
## Print value:
![image](https://github.com/user-attachments/assets/6e54bdc7-aa31-4162-891f-afcfe113e602)
## Y Prediction Value
![image](https://github.com/user-attachments/assets/b6f6bc72-0cd4-4121-8f09-578dee5d33f9)
## Confusion array
![image](https://github.com/user-attachments/assets/c84babc2-117e-437b-8a90-c4d5a4ccd8a9)
## Classification Report
![image](https://github.com/user-attachments/assets/22500e04-3a63-4256-ab94-b6468e52c10b)
## Prediction of LR
![image](https://github.com/user-attachments/assets/d4bf6f7b-6106-4e7f-a98c-abafe98bd04e)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
