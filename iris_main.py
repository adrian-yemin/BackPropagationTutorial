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

X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.2, random_state=41)

# Network and associated parameters
learningRate = 0.5
numNeuronsPerLayer = [5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3]
actFunctions = [SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1 = Network(4, numNeuronsPerLayer, actFunctions)

backprop = BackPropagationMethods()
backprop.BatchGradientDescent(n1, X_train, y_train, learningRate, 250)

num_samples = 0
num_correct_predictions = 0
for i in range(len(X_test)):
    result = n1.Output(X_test[i])
    answer_index = None
    predicted_answer_index = None
    closest_distance = None
    for j in range(len(result)):
        distance = abs(result[j][0] - 1)
        if y_test[i][j] == 1:
            answer_index = j
        if predicted_answer_index is None or closest_distance > distance:
            predicted_answer_index = j
            closest_distance = distance
    if predicted_answer_index == answer_index:
        num_correct_predictions += 1
    num_samples += 1

percent_accuracy = (num_correct_predictions/num_samples)*100
print(f"The accuracy of the neural network is: {percent_accuracy}%")
