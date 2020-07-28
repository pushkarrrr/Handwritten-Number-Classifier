import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv').as_matrix()

X = data[0:21000, 1:]
y = data[0:21000, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

count = 0

print(y_pred[1])

for i in range(0, 4200):
    count +=1 if y_pred[i]==y_test[i] else 0
    
accuracy = count/4200*100

print(accuracy)    
    
#visualising a number
temp = X_test[7, :]
y_temp = y_test[7]
temp.shape=(28, 28)
plt.imshow(255-temp, cmap='gray')
plt.show()