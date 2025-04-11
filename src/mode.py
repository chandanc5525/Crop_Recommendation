from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_model(model_type="RandomForest"):
    if model_type == "RandomForest":
        return RandomForestClassifier()
    elif model_type == "DecisionTree":
        return DecisionTreeClassifier()
    else:
        raise ValueError("Unsupported model type")
