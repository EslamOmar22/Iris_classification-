import numpy as np
from  sklearn import datasets as ds
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tkinter import *


def load_data(c1, c2, f1, f2):

    c3 = 6 - (c1 + c2)
    data = np.genfromtxt("Iris Data.txt", delimiter=',')
    iris_features = np.concatenate ((np.concatenate ((data[1:31 , [f1,f2]] , data[51:81 , [f1,f2]])) , data[101:131 , [f1,f2]]))
    iris_features_test = np.concatenate ((np.concatenate ((data[31:51 , [f1,f2]] , data[81:101 , [f1,f2]])) , data[131:151 , [f1,f2]]))

    mx=[]
    mx.append(max(iris_features[:,0]))
    mx.append(max(iris_features[:, 1]))
    mx.append(max(iris_features_test[:, 0]))
    mx.append(max(iris_features_test[:, 1]))

    iris_features[:, 0]=iris_features[:, 0]/mx[0]
    iris_features[:, 1] = iris_features[:, 1]/mx[1]
    iris_features_test[:, 0] = iris_features_test[:, 0] /mx[2]
    iris_features_test[:, 1] = iris_features_test[:, 1] /mx[3]
    iris_labels = np.zeros((60, 1))
    iris_labels_test = np.zeros((40, 1))
    iris_labels[0:30] = 1
    iris_labels[30:60] = -1
    iris_labels_test[0:20] = 1
    iris_labels_test[20:40] = -1
    if c3 == 1:
        y = np.concatenate ((iris_features[30:60 , :] , iris_features[60:90, :]))
        d = np.concatenate ((iris_features[20:40 , :] , iris_features[40:60, :]))

    elif c3 == 2:
        y = np.concatenate ((iris_features[0:30 , :] , iris_features[60:90 , :]))
        d = np.concatenate ((iris_features[0:20 , :] , iris_features[40:60, :]))

    elif c3 == 3:
        y = np.concatenate ((iris_features[0:30 , :] , iris_features[30:60 , :]))
        d = np.concatenate ((iris_features[0:20 , :] , iris_features[20:40, :]))

    return y, iris_labels, d, iris_labels_test,mx


def initialize_parameters(check):
    np.random.seed(0)
    W = np.zeros([3,1],float)
    W[0]=np.random.uniform(-1, 1)
    W[1] = np.random.uniform(-1, 1)
    W[2] = 1
    b = check
    return W, b


def confusion(predicted, real):
    con = confusion_matrix(real, predicted)
    print(con)
    acc = 0
    for i in range(2):
        acc += con[i, i]
    return (acc/len(real))*100


def train(w, b, train_x, d, epoches, learning_rate,mse_thresh,check):
    MSE = np.zeros([epoches, 1])
    epoch = np.zeros([epoches, 1])

    y = np.zeros([len(train_x), 1])
    error_ = np.zeros([len(train_x),1])
    for i in range(epoches):
        for j in range(len(train_x)):
            x = np.zeros([2, 1])
            x[0] = train_x[j, 0]
            x[1] = train_x[j, 1]

            v = (w[0]*x[0]+w[1]*x[1]) + b*w[2]

            y[j] = v
            error_[j] = (d[j]-y[j])
            w[0] = w[0] + (learning_rate*x[0]*error_[j])
            w[1] = w[1] + (learning_rate * x[1] * error_[j])
            w[2] = w[2] + (learning_rate*error_[j])*check
        mse = 0.5 * np.mean((np.power(error_, 2)))

        MSE[i]= mse
        epoch[i]=i
        print ("after " + str (i) + ' epochs MSE = ' + str (mse))
        if mse <= mse_thresh:
            break
    return w, b,epoch,MSE


def test (W, B, test_x, d):
    y = np.zeros([len(test_x), 1])
    b=B
    w=W
    for i in range(len(test_x)):
        x = np.zeros([2, 1])
        x[0] = test_x[i, 0]
        x[1] = test_x[i, 1]
        v = (w[0] * x[0] + w[1] * x[1]) + b * w[2]
        y[i]=np.sign(v)

    print("test accuarcy is   ", confusion(y, d), '  %')


def draw_line(w):
    y=[]
    I=[]
    b=w[2]
    for i in np.linspace(0.2,1):
        slope = -(b/w[0])/(b/w[1])
        intercept = -b/w[0]
        if(b==1):
            intercept=0
        y.append((slope*i) + intercept)
        I.append(i)

    return y,I


def execute():
    c_1 = int(c1.get())
    c_2 = int(c2.get())
    f_1 = int(f1.get())
    f_2 = int(f2.get())
    epochs = int(epc.get())
    mse_threshold = float(mse.get())
    learning_rate = float(LR.get())
    check_bias = bb.get()

    check_bias= (check_bias == 'y')

    train_x, train_y, test_x, test_y,mx = load_data(c_1, c_2, f_1-1, f_2-1)
    w, b  = initialize_parameters(check_bias)
    weight, bias,epoch,MSE = train(w, b, train_x, train_y, epochs, learning_rate, mse_threshold,check_bias)

    test(weight, bias, test_x, test_y)

    y2,i2 = draw_line(weight)


    for X,d in zip(test_x,test_y):
        plt.plot(X[0],X[1],'ro' if (d == 1.0) else 'bo')
    plt.plot(y2, i2)
    plt.xlabel("X ")
    plt.ylabel("Y")
    plt.title("LSM")
    plt.show()
    plt.figure()
    plt.plot(epoch, MSE)
    plt.xlabel("epoch number ")
    plt.ylabel("MSE")
    plt.title("Learning Curve")
    plt.show()

    return


if __name__ == '__main__':
    top = Tk()
    ment = StringVar()
    button = Button(top, text="  Execute  ", command=execute)
    button.grid(row=28, column=3)
    top.geometry("250x250")
    top.title('TASK__2')
    label=Label(top,text="class_1")
    label.grid(row=10,column=2)
    c1 = Entry(top,textvariable=StringVar())
    c1.insert(0,"1")
    c1.grid(row=10, column=3)
    label = Label(top, text="class_2")
    label.grid(row=12, column=2)
    c2 = Entry(top,textvariable=StringVar())
    c2.insert(0, "3")
    c2.grid(row=12, column=3)
    label = Label(top, text="Feature_1")
    label.grid(row=14, column=2)
    f1 = Entry(top,textvariable=StringVar())
    f1.insert(0, "1")
    f1.grid(row=14, column=3)
    label = Label(top, text="Feature_2")
    label.grid(row=16, column=2)
    f2= Entry(top,textvariable=StringVar())
    f2.insert(0, "2")
    f2.grid(row=16, column=3)
    label = Label(top, text="Epochs")
    label.grid(row=18, column=2)
    epc = Entry(top,textvariable=StringVar())
    epc.insert(0, "2000")
    epc.grid(row=18, column=3)
    label = Label(top, text="mse_threshold")
    label.grid(row=20, column=2)
    mse = Entry(top,textvariable=StringVar())
    mse.insert(0, "0.01")
    mse.grid(row=20, column=3)
    label = Label(top, text="learning_rate")
    label.grid(row=22, column=2)
    LR = Entry(top,textvariable=StringVar())
    LR.insert(0, "0.01")
    LR.grid(row=22, column=3)
    label = Label(top, text="check_bias")
    label.grid(row=24, column=2)
    bb = Entry(top,textvariable=StringVar())
    bb.insert(0, "y")
    bb.grid(row=24, column=3)

    top.mainloop()

