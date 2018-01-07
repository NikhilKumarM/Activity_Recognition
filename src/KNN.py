import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

# getting the data
print("----------------------------------- Implementing K Nearest Neighbours -------------------------------------------------------")
total_data = pd.read_csv('../PreProccessed_Data_Window_Sizewise/W500/Train_Data.csv')
total_label = pd.read_csv('../PreProccessed_Data_Window_Sizewise/W500/Train_Label.csv',header=None)
total_label = total_label.astype('int')
total_data = total_data.fillna(0)


# normalizing the training data
total_data = normalize(total_data,axis=0)
total_label = total_label.values.ravel()

print("Total Data ")
print(total_data.shape)
print("Total Data label ")
print(total_label.shape)

print("Dividing the data to train and test ")
X_train, X_test, y_train, y_test = train_test_split(total_data, total_label, test_size=0.20, random_state=1)

print("Dimensions of train data ")
print(X_train.shape)
print("Dimensions of training labels ")
print(y_train.shape)
print("Dimensions of test data ")
print(X_test.shape)
print("Dimensions of test labels ")
print(y_test.shape)


print(" ----------------Training KNeighborsClassifier Model --------------")
classifier = KNeighborsClassifier(n_jobs=-1,n_neighbors= 5)
classifier.fit(X_train, y_train)

print("-------------------------Testing the Model------------------------")
label_predicted = classifier.predict(X_test)

accuracy = accuracy_score(y_test, label_predicted)

print("Test Accuracy ")
print(accuracy * 100)


