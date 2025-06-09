import numpy as np                                                                                                            
import pandas as pd                                                
import seaborn as sns                                
import matplotlib.pyplot as plt                                           
from   sklearn.preprocessing import StandardScaler                               
from   sklearn.model_selection import train_test_split                                                       
from   sklearn.feature_extraction.text import TfidfVectorizer                                            
                                                                                                             
                                                                                                    
sonar=pd.read_csv(r"D:\sonar data.csv" , header=None)                                               
print(sonar.head())                                                  
print(sonar.shape)                                                              
print(sonar.info())                                                                          
print(sonar.describe())                                                        
                                                                                                                   
# in that dataset there is no t
# so label encoding wiil be applied 
# no more text data pre-processing 
# data standardization , checking and fixing imbalanced data , train text split

# checking no of rock and mine 

rm=sonar[60].value_counts()
print(rm)

# m=111 , r=97 no need to fix imbalanced data 

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
sonar['Target']=le.fit_transform(sonar[60])
sonar=sonar.drop(columns=60 , axis=1)
print(sonar.head())


# train , test , split and standardization of data 
X=sonar.drop(columns='Target' , axis=1)
Y=sonar['Target']

X_train , X_test , y_train , y_test=train_test_split(X,Y , test_size=0.1 , stratify=Y , random_state=2)
print(X.shape , X_train.shape , X_test.shape)
      
scaler=StandardScaler()

X_trainsta=scaler.fit_transform(X_train)
X_teststa=scaler.transform(X_test)

print(X_trainsta.std(), X_teststa.mean())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression()
model.fit(X_trainsta , y_train)

X_prediction=model.predict(X_trainsta)
Train_accuracy=accuracy_score(X_prediction , y_train)

print("accuracy of train data",Train_accuracy)

x_testprediction=model.predict(X_teststa)

test_accuracy=accuracy_score(x_testprediction , y_test)
print("accuracy score of test data" , test_accuracy)
