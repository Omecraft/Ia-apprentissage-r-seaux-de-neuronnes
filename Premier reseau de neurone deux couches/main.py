import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
import utilities

import time
import tqdm

def normalyse(X):
    #( passer de 0 a 255 a 0 a 1)
    return (X - X.min()) / (X.max() - X.min())

def flatten(X):
    return X.reshape(X.shape[0], -1)
def log_loss(y, A):
    eps = 1e-15
    A = np.clip(A, eps, 1 - eps)
    return 1/len(y) * np.sum(-y * np.log(A) - (1-y) * np.log(1-A))

def initialisation(n0,n1,n2):
    W1 = np.random.randn(n1,n0)
    b1 = np.random.randn(n1,1)
    W2 = np.random.randn(n2,n1)
    b2 = np.random.randn(n2,1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    Z1= parameters["W1"].dot(X) + parameters["b1"]
    A1= 1 / (1 + np.exp(-Z1))
    Z2= parameters["W2"].dot(A1) + parameters["b2"]
    A2= 1 / (1 + np.exp(-Z2))

    activations = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return activations

def back_propagation(X, y, activations, parameters):
    A1= activations["A1"]
    A2= activations["A2"]
    W2= parameters["W2"]
    m = X.shape[1]
    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

def update(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    A2 = activations["A2"]
    return A2 > 0.5

def neural_network(X_train,y_train,n1,learning_rate=0.1,n_iter=1000):
    n0 =X_train.shape[0]
    n2=y_train.shape[0]
    parametres = initialisation(n0,n1,n2)
    train_loss=[]
    train_acc=[]
    true_accuracy=[]
    for i in tqdm.tqdm(range(n_iter)):
        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, activations, parametres)
        parametres = update(parametres, gradients, learning_rate)
        if i % 100 == 0:
            
            loss = log_loss(y_train, activations["A2"])
            train_loss.append(loss)
            y_pred = predict(X_train, parametres)
            accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_acc.append(accuracy)
            true_accuracy.append(accuracy_score(y_test.flatten(), predict(X_test, parametres).flatten()))


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_loss)
    plt.title("Loss")
    plt.subplot(1, 3, 2)
    plt.plot(train_acc)
    plt.title("Accuracy")
    # Frontiere de decision( avec le trait)
    plt.subplot(1, 3, 3)
    #plt.scatter(X_train[0, y_train.flatten()==0], X_train[1, y_train.flatten()==0], 
                #c='red', edgecolors='k', label='Classe 0', s=50)
    #plt.scatter(X_train[0, y_train.flatten()==1], X_train[1, y_train.flatten()==1], 
                #c='blue', edgecolors='k', label='Classe 1', s=50)
    
    # Calcul de la vraie frontière de décision (A=0.5)
    #h = 0.02
    #x_min, x_max = X_train[0, :].min() - 0.5, X_train[0, :].max() + 0.5
    #y_min, y_max = X_train[1, :].min() - 0.5, X_train[1, :].max() + 0.5
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Prédiction sur la grille
    #grid_points = np.c_[xx.ravel(), yy.ravel()].T
    #Z = predict(grid_points, parametres)
    #Z = Z.reshape(xx.shape)
    
    # Tracer la ligne de frontière (contour où A=0.5)
    #plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=3)
    
    #plt.title("Classification avec frontière de décision (A=0.5)")
    
    
    #plt.xlabel("X1")
    #plt.ylabel("X2")
    #plt.legend()
    plt.plot(true_accuracy)
    plt.title("True Accuracy")

    plt.show()
    
        
    return parametres

def test_circle():
    X,y = make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=0)
    X = X.T
    y = y.reshape(1, y.shape[0])
    return X, y

X_train, y_train, X_test, y_test = utilities.load_data()
X_train = flatten(X_train)
X_test = flatten(X_test)
X_train = normalyse(X_train)
X_test = normalyse(X_test)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

print("dimension de X_train:", X_train.shape)
print("dimension de X_test:", X_test.shape)
print("Max de X_train:", X_train.max())
print("Min of X_train:", X_train.min())
print("Max of X_test:", X_test.max())
print("Min of X_test:", X_test.min())







parametres = neural_network(X_train, y_train, n1=32, learning_rate=0.1, n_iter=30000)
