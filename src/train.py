from sklearn.svm import SVR

def train_svr_model(X_train_scaled, y_train):
    model = SVR(kernel='rbf', C=100.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    return model