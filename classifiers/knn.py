from sklearn.neighbors import KNeighborsClassifier

def train(x_train, y_train, params):
    knn = KNeighborsClassifier(n_neighbors=params['k'])
    knn.fit(x_train, y_train)

    return knn

def classify(model, test):
    y_pred = model.predict(test)

    return y_pred