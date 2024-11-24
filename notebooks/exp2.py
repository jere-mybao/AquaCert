# Import libraries
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
from sklearn.model_selection import train_test_split

# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(repo_owner="jere-mybao", repo_name="mlops", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/jere-mybao/mlops.mlflow")

# Load the dataset from CSV file
data = pd.read_csv(
    "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/refs/heads/main/water_potability.csv",
    sep=",",
)

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


# Define a function to fill missing values with the median
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df


# Fill missing values in both the training and test datasets
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Split the data into features (X) and target (y)
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define multiple baseline models to compare performance
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
    "SVC": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "XGBClassifier": XGBClassifier(),
}

# Iterate over each model in the dictionary
for model_name, model in models.items():
    # Start a new MLflow run for each model
    with mlflow.start_run(run_name=model_name):
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Save the trained model using pickle
        model_filename = f"{model_name}.pkl"
        pickle.dump(model, open(model_filename, "wb"))

        # Load the saved model for predictions
        loaded_model = pickle.load(open(model_filename, "rb"))
        y_pred = loaded_model.predict(X_test)

        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model parameters to MLflow
        if hasattr(model, "get_params"):
            params = model.get_params()
            mlflow.log_params(params)

        # Generate and save the confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_filename)
        plt.close()

        # Log the confusion matrix image to MLflow
        mlflow.log_artifact(cm_filename)

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, model_name)

        # Handle __file__ safely
        try:
            mlflow.log_artifact(__file__)
        except NameError:
            print("The __file__ variable is not defined; skipping source code logging.")

        # Set tags in MLflow
        mlflow.set_tag("author", "jeremy-bao")

        # Print out the performance metrics for reference
        print(f"Model: {model_name}")
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("-" * 30)

print("All models have been trained and logged successfully.")
