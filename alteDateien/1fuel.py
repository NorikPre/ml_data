import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Daten laden und vorbereiten
data = pd.read_csv("autos.csv")
data = data.drop(["HWY (L/100 km)", "COMB (L/100 km)", "COMB (mpg)", "EMISSIONS"], axis=1)
X = data.drop("FUEL CONSUMPTION", axis=1)
y = data["FUEL CONSUMPTION"]
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2143, random_state=None)
# Vorverarbeitungspipeline
categorical_features = ["MAKE", "MODEL", "VEHICLE CLASS", "TRANSMISSION", "FUEL"]
numerical_features = ["YEAR", "ENGINE SIZE", "CYLINDERS"]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Funktion zum Trainieren von Modellen mit MLflow
def train_model_with_mlflow(params, X_train, y_train, X_val, y_val):
    experiment_name = "Fuel Consumption Prediction 2"
    client = MlflowClient()
    experiment_id = client.create_experiment(experiment_name) if client.get_experiment_by_name(experiment_name) is None else client.get_experiment_by_name(experiment_name).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor(random_state=422, **params))])
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model_pipeline, "model")

# Liste der Hyperparameter-Kombinationen
hyperparameters_list = [
    {'n_estimators': 10, 'max_depth': 45},
]

# Modelle mit verschiedenen Hyperparametern trainieren
for params in hyperparameters_list:
    train_model_with_mlflow(params, X_train, y_train, X_val, y_val)

# Funktion zum Abrufen des besten Modells
def get_best_model(experiment_id, metric_name='mse', ascending=True):
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=[f"metric.{metric_name} {'asc' if ascending else 'desc'}"])
    best_run = runs[0]
    return mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")

# Experiment-ID abrufen und bestes Modell laden
experiment = mlflow.get_experiment_by_name("Fuel Consumption Prediction 2")
experiment_id = experiment.experiment_id

# Bestes Modell laden
best_model = get_best_model(experiment_id)

# Vorhersagen mit dem besten Modell treffen
y_pred = best_model.predict(X_test)

# Modellleistung ausgeben
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error des besten Modells:", mse)
