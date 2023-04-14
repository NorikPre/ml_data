import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Daten laden und vorbereiten
data = pd.read_csv("autos.csv")
data = data.drop(["HWY (L/100 km)", "COMB (L/100 km)", "COMB (mpg)", "EMISSIONS"], axis=1)
X = data.drop("FUEL CONSUMPTION", axis=1)
y = data["FUEL CONSUMPTION"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Vorverarbeitungspipeline
categorical_features = ["MAKE", "MODEL", "VEHICLE CLASS", "TRANSMISSION", "FUEL"]
numerical_features = ["YEAR", "ENGINE SIZE", "CYLINDERS"]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# K-Nearest Neighbors Regressor erstellen
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # Hier können Sie die Anzahl der Nachbarn (n_neighbors) anpassen

# Pipeline erstellen
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', knn_regressor)])

# MLflow-Experiment erstellen
experiment_name = "KNN Fuel Consumption Prediction"
client = MlflowClient()
experiment_id = client.create_experiment(experiment_name) if client.get_experiment_by_name(experiment_name) is None else client.get_experiment_by_name(experiment_name).experiment_id

# MLflow-Run starten
with mlflow.start_run(experiment_id=experiment_id):
    # Modell trainieren
    model_pipeline.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = model_pipeline.predict(X_test)

    # Modellleistung ausgeben
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error des K-Nearest Neighbors Regressors:", mse)

    # MLflow-Metriken und -Modelle protokollieren
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model_pipeline, "knn_regressor")
