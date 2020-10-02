import math
import numpy as np
import sklearn
import pandas as pd



df = pd.read_csv('breast-cancer-wisconsin_data.txt')
df.replace('?',-99999, inplace=True)


X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test =sklearn.model_selection.train_test_split(X, y, test_size=0.2)
clf =sklearn.neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
example_measures = np.array([4,2,1,1,1,2,3,2,1])
prediction = clf.predict(example_measures)
print(prediction)
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)

example_measures =np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(2, -1)
prediction = clf.predict(example_measures)
print(prediction)

example_measures =np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures =example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = math.sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-
plot2[1])**2 )
