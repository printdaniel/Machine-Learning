import os
import sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

plt.style.use("ggplot")

iris = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data")
data = pd.read_csv(iris, header=None, encoding="utf-8")

proc = preprocessing.LabelEncoder()
sepal_length = proc.fit_transform(list(data[0]))
sepal_width = proc.fit_transform(list(data[1]))
petal_length = proc.fit_transform(list(data[2]))
petal_width = proc.fit_transform(list(data[3]))
species = proc.fit_transform(list(data[4]))

predict = 4 # Species 

x = list(zip(sepal_length, sepal_width, petal_length, petal_width))
y = list(species)


class PredictionAndAccurracies(object):
    def __init__(self):
        self.results = []
        self.accuracies = []
        self.best = 0
        self.worst = 100
        self.total_acc = 0

        self.prediction()
        self.accumulate_accuracies()
        self.print_predictions()

    def __str__(self):
        return str(f"Best Accuracy:    {round((self.best * 100), 2)}%\n"\
                   f"Worst Accuracy:   {round((self.worst * 100), 2)}%\n"\
                   f"Average Accuracy: {round((self.total_acc * 100), 2)}%\n")


    def prediction(self):
        """ Populates list of the results produced for each prediction """
        self.variations = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    
        for i in range(100):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.7)
            
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            self.accuracies.append(accuracy)

            if accuracy > self.best:
                self.best = accuracy
            elif accuracy < self.worst:
                self.worst = accuracy
            
            prediction = model.predict(x_test)
            self.results.append(f"Prediction:\t{self.variations[prediction[i]].ljust(10)}"\
                                f"\t\tActual: {self.variations[y_test[i]].ljust(10)}"\
                                f"\t\tAccuracy: {str(round(accuracy * 100, 2)).ljust(5)}%"\
                                f"\tData: {x_test[i]}")        


    def accumulate_accuracies(self):
        """ Calculates the total accuracy of the predictions minus the 2 fist and last predictions """
        del self.accuracies[:3]
        del self.accuracies[97:]

        for acc in self.accuracies:
            self.total_acc += acc

        self.total_acc = self.total_acc / len(self.accuracies)


    def print_predictions(self):
        for result in self.results:
            print(result)


print(PredictionAndAccurracies())


y = data.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = data.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="green", marker="x", label="Versicolor")
plt.title("Linearly Separable Data")
plt.xlabel("Sepal length in cm")
plt.ylabel("Petal length in cm")
plt.grid(True)
plt.legend()
