import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import time
import utilities
import tqdm

def normalyse(X):
    #( passer de 0 a 255 a 0 a 1)
    return (X - X.min()) / (X.max() - X.min())

def flatten(X):
    return X.reshape(X.shape[0], -1)

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    print("W:", W.shape)
    print("b:", b.shape)
    return (W, b)

def model(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def logloss(A, y):
    eps = 1e-15
    A = np.clip(A, eps, 1 - eps)
    return 1/len(y) * np.sum(-y * np.log(A) - (1-y) * np.log(1-A))

def gradients(A, X, y):
    dw = 1/len(y) * np.dot(X.T, (A-y))
    db = 1/len(y) * np.sum(A-y)
    return (dw, db)

def update(W, b, dw, db, learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    return (W, b)

def artificial_neuron(X, y, learning_rate=0.01, num_iterations=1000):
    W, b = initialisation(X)
    losses = []
    accuracies = []
    true_accuracies = []
    start_time = time.time()
    esti = time.time()
    for i in tqdm.tqdm(range(num_iterations)):
        A = model(X, W, b)
        
        dw, db = gradients(A, X, y)
        W, b = update(W, b, dw, db, learning_rate)
        
        if i % 100 == 0 and i != 0:
            #estimated_time = time.time() - esti
            #due = num_iterations - i
            #due_time = due * estimated_time
            #due_time = time.strftime("%H:%M:%S", time.gmtime(due_time))
            #print(f"Estimated time: {due_time}")
            loss = logloss(A, y)
            losses.append(loss)
            accuracy = accuracy_score(y, A > 0.5)
            accuracies.append(accuracy)
            y_pred = predict(X_test, W, b)
            true_accuracy = accuracy_score(y_test, y_pred > 0.5)
            true_accuracies.append(true_accuracy)

            
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Loss")
    plt.subplot(1, 3, 2)
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.subplot(1, 3, 3)
    plt.plot(true_accuracies)
    plt.title("True Accuracy")
    plt.show()
    y_pred = predict(X, W, b)
    accuracy = accuracy_score(y, y_pred > 0.5)
    print(f"Accuracy: {accuracy}")
    
    end_time = time.time()
    print(f"Temps d'execution: {end_time - start_time} secondes")
    return (W, b, accuracy)

def predict(X, W, b):
    A = model(X, W, b)
    return A



X_train, y_train, X_test, y_test = utilities.load_data()

print("dimension de X_train:", X_train.shape)
print("dimension de y_train:", y_train.shape)
print("dimension de X_test:", X_test.shape)
print("dimension de y_test:", y_test.shape)

print("Max de X_train:", X_train.max())
print("Min of X_train:", X_train.min())
print("Max of X_test:", X_test.max())
print("Min of X_test:", X_test.min())

X_train = flatten(X_train)
X_test = flatten(X_test)
X_train = normalyse(X_train)
X_test = normalyse(X_test)


print("dimension de X_train:", X_train.shape)
print("dimension de X_test:", X_test.shape)
print("Max de X_train:", X_train.max())
print("Min of X_train:", X_train.min())
print("Max of X_test:", X_test.max())
print("Min of X_test:", X_test.min())



W, b, accuracy = artificial_neuron(X_train, y_train, learning_rate=0.01, num_iterations=100000)
print("Accuracy:", accuracy)
y_pred = predict(X_test, W, b)
accuracy = accuracy_score(y_test, y_pred > 0.5)
print("Accuracy:", accuracy)
print("Fin")
