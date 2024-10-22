from Network import Network
from SigmoidActivation import SigmoidActivation
import numpy
from BackPropagationMethods import BackPropagationMethods
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/adrian/Downloads/iris2.csv')

X = data.drop(['Setosa', "Versicolor", "Virginica"], axis=1)
y = data.drop(['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], axis=1)

X = X.to_numpy()
y = y.to_numpy()
y_list = y.tolist()
X_list = X.tolist()
for i in range(len(X_list)):
    X_list[i] = numpy.array(X_list[i])
    X_list[i] = numpy.reshape(X_list[i], (4, 1))

for j in range(len(y_list)):
    y_list[j] = numpy.array(y_list[j])
    y_list[j] = numpy.reshape(y_list[j], (3, 1))
    y_list[j] = y_list[j].tolist()

X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.2, random_state=41)

# Network and associated parameters
learningRate = 0.05
numNeuronsPerLayer = [5, 3]
actFunctions = [SigmoidActivation(), SigmoidActivation()]
n1 = Network(4, numNeuronsPerLayer, actFunctions)

backprop = BackPropagationMethods()
num_samples = 0
num_correct_predictions = 0
for epoch in range(100):
    backprop.StochasticGradientDescent(n1, X_train, y_train, learningRate, 100)
    for i in range(len(X_test)):
        result = n1.Output(X_test[i])
        print(X_test[i])
        print(result)
        print(y_test[i])
        _max = -1
        max_dex = -1
        ans_max = -1
        ans_max_dex = -1
        for j in range(len(result)):
            if result[j][0] > _max:
                _max = result[j][0]
                max_dex = j
            if y_test[i][j][0] > ans_max:
                ans_max = y_test[i][j][0]
                ans_max_dex = j
        if max_dex == ans_max_dex:
            num_correct_predictions += 1
        num_samples += 1

percent_accuracy = (num_correct_predictions/num_samples)*100
print(f"The accuracy of the neural network is: {percent_accuracy}%")
