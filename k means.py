import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report
#load dataset
letter = pd.read_csv('letter.csv')
#View missing values across entire dataset
letter.isnull().sum().sum()
print("Dimensions: ", letter.shape, "\n")
print(letter.info())
print(letter.describe())
#sorting alphabetical order
order = list(np.sort(letter['letter'].unique()))
print(order)
# Find mean of each feature grouped by letter
#View results
letter_means = letter.groupby('letter').mean()
letter_means.head(26)
#plotting heatmaps
plt.figure(figsize=(18, 10))
sns.heatmap(letter_means)
# mean of feature values
new = round(letter.drop('letter', axis=1).mean(), 2)
new.view()
new.hist()
#Letter counts
sns.countplot(letter.iloc[:,0],order = order)

#pixel values range from 0 to 255
#set independent variable values to this range 
#finding the minimum and maximum values for each independent variable would esily identify any outliers 
print(letter.min())
print(letter.max())

#Next  thing is to split our data into training and test data 
#Then we move on to build our models 
# Then we evalaute the accuracy of our model
y = new1.letter 
X = new1.drop(['letter'], axis =1 )
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
log_reg.score (X_test, y_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#Plot confusion matrix 
cm = confusion_matrix(y_train, log_reg.predict(X_train))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Bs', 'Predicted Ds'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Bs', 'Actual Ds'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()
#Next  thing is to split our data into training and test data 
#Then we move on to build our models 
# Then we evalaute the accuracy of our model
y = new2.letter 
X = new2.drop(['letter'], axis =1 )
X.head()
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#create logistic classifier 
log_reg = LogisticRegression()
#Fit log_reg to training data 
log_reg.fit(X_train,y_train)
#predict with our test set
y_pred = log_reg.predict(X_test)
#compute accuracy
log_reg.score (X_test, y_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#Plot confusion matrix 
cm = confusion_matrix(y_train, log_reg.predict(X_train))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Es', 'Predicted Fs'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Es', 'Actual Fs'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
