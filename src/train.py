import logging
import joblib
import yaml
from src.data_loader import load_data, split_data
from src.preprocessing import scale_features
from src.model import get_model
from src.evaluate import evaluate_model

def load_config(path='configs/config.yaml'):
    with open(path) as file:
        return yaml.safe_load(file)

def main():
    logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
    
    config = load_config()

    df = load_data(config["data"]["url"])
    X_train, X_test, y_train, y_test = split_data(df, config["data"]["test_size"], config["data"]["random_state"])
    
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, config["scaling"])
    
    model = get_model(config["model"]["type"])
    model.fit(X_train_scaled, y_train)
    
    evaluate_model(model, X_test_scaled, y_test)

    joblib.dump(model, config["model"]["save_path"])
    logging.info(f"Model saved to {config['model']['save_path']}")

if __name__ == "__main__":
    main()
