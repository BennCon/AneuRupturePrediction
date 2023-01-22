from operator import itemgetter
from sklearn.linear_model import LogisticRegression

  
def train(x_train, y_train, params):
    solver, max_iter, c = itemgetter('solver', 'max_iter', 'C')(params)
    #Fit model
    lr = LogisticRegression(solver=solver, max_iter=max_iter, C=c)
    lr.fit(x_train, y_train)

    return lr

def classify(model, test_data):
    #Predict on test data
    y_pred = model.predict(test_data)

    return y_pred
    