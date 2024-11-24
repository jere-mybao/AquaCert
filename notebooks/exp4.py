# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import dagshub

# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(repo_owner="jere-mybao", repo_name="mlops", mlflow=True)
mlflow.set_experiment("Experiment 4")
mlflow.set_tracking_uri("https://dagshub.com/jere-mybao/mlops.mlflow")

# Load and preprocess data
data = pd.read_csv(
    "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/refs/heads/main/water_potability.csv",
    sep=",",
)

# Split the dataset into training and test sets with an 80-20 split
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


# Function to fill missing values with the mean of each column
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
    return df


# Apply the function to fill missing values in both training and test sets
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# Prepare the training data by separating features and target variable
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Define the Random Forest Classifier model and the parameter grid for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [None, 4, 5, 6, 10],
}

# Use GridSearchCV to evaluate all combinations
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

# Start a new MLflow run to log the Random Forest tuning process
with mlflow.start_run(run_name="Random Forest Grid Search"):
    # Fit the GridSearchCV object on the training data
    grid_search.fit(X_train, y_train)

    # Save cv_results_ to a DataFrame
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv("grid_search_cv_results.csv", index=False)
    mlflow.log_artifact("grid_search_cv_results.csv")

    # Print the best hyperparameters found by GridSearchCV
    print("Best parameters found: ", grid_search.best_params_)

    # Log the best parameters in MLflow
    mlflow.log_params(grid_search.best_params_)

    # Train the model using the best parameters
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # Save the trained model to a file
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Prepare the test data
    X_test = test_processed_data.drop(columns=["Potability"], axis=1)
    X_test = scaler.transform(X_test)
    y_test = test_processed_data["Potability"]

    # Load the saved model
    model = pickle.load(open("model.pkl", "rb"))

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log performance metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save and log training and test data
    train_processed_data.to_csv("train_data.csv", index=False)
    test_processed_data.to_csv("test_data.csv", index=False)
    mlflow.log_artifact("train_data.csv")
    mlflow.log_artifact("test_data.csv")

    # Handle __file__ safely
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        print("The __file__ variable is not defined; skipping source code logging.")

    # Infer the model signature
    sign = infer_signature(X_test, model.predict(X_test))

    # Log the trained model in MLflow
    mlflow.sklearn.log_model(model, "Best_Model", signature=sign)

    # Print performance metrics
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)

print("Model training and logging completed successfully.")
