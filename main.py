import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def draw_iris_data():
    data = np.genfromtxt("Iris Data.txt", delimiter=',')
    plt.plot(data[:50, 0], data[:50, 1], 'ro')
    plt.plot(data[51:100, 0], data[51:100, 1], 'go')
    plt.plot(data[101:, 0], data[101:, 1], 'bo')
    plt.show()
    plt.plot (data[:50, 1], data[:50, 3], 'ro')
    plt.plot (data[51:100, 1], data[51:100, 3], 'go')
    plt.plot (data[101:, 1], data[101:, 3], 'bo')
    plt.show ()
    plt.plot (data[:50 , 2] , data[:50 , 3] , 'ro')
    plt.plot (data[51:100 , 2] , data[51:100 , 3] , 'go')
    plt.plot (data[101: , 2] , data[101: , 3] , 'bo')
    plt.show ()
    plt.plot (data[:50 , 0] , data[:50 , 3] , 'ro')
    plt.plot (data[51:100 , 0] , data[51:100 , 3] , 'go')
    plt.plot (data[101: , 0] , data[101: , 3] , 'bo')
    plt.show ()
    plt.plot (data[:50 , 0] , data[:50 , 2] , 'ro')
    plt.plot (data[51:100 , 0] , data[51:100 , 2] , 'go')
    plt.plot (data[101: , 0] , data[101: , 2] , 'bo')
    plt.show ()


def load_dataset(c1, c2, f1, f2):
    f1 -= 1
    f2 -= 1
    c3 = 6 - (c1 + c2)
    data = np.genfromtxt("Iris Data.txt", delimiter=',')
    iris_features = np.concatenate ((np.concatenate ((data[0:30 , :4] , data[50:80 , :4])) , data[100:130 , :4]))
    iris_features_test = np.concatenate ((np.concatenate ((data[30:50 , :4] , data[80:100 , :4])) , data[130:150 , :4]))
    iris_features_test = np.delete(iris_features_test, [f1, f2], axis=1)
    iris_features = np.delete(iris_features, [f1, f2], axis=1)
    iris_labels = np.zeros((60, 1))
    iris_labels_test = np.zeros((40, 1))
    iris_labels[:30] = 1
    iris_labels[30:60] = -1
    iris_labels_test[:20] = 1
    iris_labels_test[20:40] = -1
    if c3 == 1:
        iris_features = np.concatenate ((iris_features[30:60 , :] , iris_features[60:90, :]))
        iris_features_test = np.concatenate ((iris_features[20:40 , :] , iris_features[40:60, :]))

    elif c3 == 2:
        iris_features = np.concatenate ((iris_features[0:30 , :] , iris_features[60:90 , :]))
        iris_features_test = np.concatenate ((iris_features[:20 , :] , iris_features[40:60, :]))

    elif c3 == 3:
        iris_features = np.concatenate ((iris_features[0:30 , :] , iris_features[30:60 , :]))
        iris_features_test = np.concatenate ((iris_features[:20 , :] , iris_features[20:40, :]))

    return iris_features, iris_labels, iris_features_test, iris_labels_test


def initialize_parameters(check):
    np.random.seed(0)
    W = np.random.randn(1, 2)*.01
    b = check*(np.random.randn(1, 1)*.01)
    return W, b


def linear_forward(X, W, b):
    Z = np.dot(W, X.T) + b
    return Z


def activation_signum(Z):
    prediction = np.sign(Z)
    return prediction


def update_weights(old_W, old_b, error_, learning_rate, train_x, check):
    new_w = old_W + (np.dot(error_, train_x)*learning_rate)
    new_b = check*(old_b + np.random.rand(1, 1)*.01)
    return new_w, new_b


def error(predicted_labels, real_label):
    error_ = real_label - predicted_labels
    return error_


def confusion(predicted, real):
    con = confusion_matrix(real, predicted)
    acc = 0
    for i in range(2):
        acc += con[i, i]
    return (acc/len(real))*100


def train(W, B, train_x, train_y, epoches, learning_rate, check):
    w = W
    b = B
    errors = []
    epoch = []
    for i in range(epoches):
        Z = linear_forward(train_x, w, b)
        predicted = activation_signum(Z)
        error_ = error(predicted, train_y.T)
        w, b = update_weights(w, b, error_, learning_rate, train_x, check)
        acc = confusion(predicted[0], train_y.T[0])
        errors.append(100-acc)
        epoch.append(i)
        print ("after " + str (i) + ' epochs    error = ' + str (100 - acc) + '%')
        if acc >= 99:
            break
    plt.plot(epoch, errors)
    plt.xlabel("epoch number ")
    plt.ylabel("error")
    plt.show()
    return w, b


def test (W, B, test_x, test_y):
        Z = linear_forward (test_x , W , B)
        predicted = activation_signum(Z)
        print("test accuarcy is   ", confusion(predicted[0], test_y.T[0]), '  %')


if __name__ == '__main__':
    draw_iris_data()
    c1 = int (input ("enter the first class :  "))
    c2 = int(input("enter the second class :  "))
    f1 = int (input ("enter the first feature you want to drop :  "))
    f2 = int (input ("enter the second feature you want to drop :  "))
    epochs = int (input ("enter number of epochs:  "))
    learning_rate = float (input ("enter the number of the learning_rate: "))
    check_bias = (input ("Do you want to add bias ? Y/N : "))
    if check_bias == 'Y':
        check_bias = 1
    else:
        check_bias = 0
    train_x , train_y, test_x, test_y = load_dataset(c1, c2, f1, f2)
    w, b = initialize_parameters(check_bias)
    weight, bias = train(w, b, train_x, train_y, epochs, learning_rate, check_bias)
    test(weight, bias, test_x, test_y)
