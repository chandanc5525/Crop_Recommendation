from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

def scale_features(X_train, X_test, method='standard'):
    logging.info(f"Scaling using: {method}")
    scaler = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }.get(method, StandardScaler())

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
