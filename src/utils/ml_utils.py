from sklearn import preprocessing

def data_transform(X, scaler_type = "Standard"):

    if scaler_type == "Standard":
        scaler = preprocessing.StandardScaler()
    elif scaler_type == "MinMax":
        scaler = preprocessing.MinMaxScaler()
    else:
        raise Exception("Scaler type can be either Standard or MinMax!")
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler