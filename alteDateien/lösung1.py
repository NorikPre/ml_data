from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pandas as pd
#from sklearn.preprocessing import StandardScaler 
#Entscheidungsbäume sind im Allgemeinen weniger empfindlich gegenüber unterschiedlichen Skalen der Merkmale
from sklearn.model_selection import train_test_split
import category_encoders as ce


# Lade die .csv-Datei als DataFrame
data = pd.read_csv("autos.csv")

# Liste der Spalten, die kodiert werden sollen
columns_to_encode = ['MAKE', 'VEHICLE CLASS', 'TRANSMISSION']

# Target Encoder erstellen
encoder = ce.TargetEncoder(cols=columns_to_encode)

# Encoder an die Daten anpassen
encoder.fit(data[columns_to_encode], data['COMB (L/100 km)'])

# Kodierung anwenden
encoded_data = encoder.transform(data[columns_to_encode])

# Namen der neu erstellten Spalten ändern
encoded_data.columns = [f"{col}_enc" for col in columns_to_encode]

# Kodierte Daten mit dem ursprünglichen DataFrame zusammenführen
data = pd.concat([encoded_data, data], axis=1)


# Umcodierung der Werte in der Spalte FUEL
data["FUEL"] = data["FUEL"].replace({'D': 1, 'X': 2, 'Z': 3, 'N': 4, 'E': 5})
# Entferne die aufgelisteten Spalten aus dem DataFrame
columns_to_remove = ["MODEL",'MAKE', 'VEHICLE CLASS', "FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (mpg)", "EMISSIONS"]
data = data.drop(columns_to_remove, axis=1)


# Beispiel: Erstelle ein neues Feature namens "ENGINE_CYLINDER_RATIO" als Verhältnis von ENGINE SIZE zu CYLINDERS
data["ENGINE_CYLINDER_RATIO"] = data["ENGINE SIZE"] / data["CYLINDERS"]

data["TRANSMISSION_TYPE"] = data["TRANSMISSION"].apply(lambda x: 0 if "A" in x else 1)
# Entferne die Spalte "TRANSMISSION"
data = data.drop("TRANSMISSION", axis=1)


# Trenne die Zielvariable von den Features
X = data.drop("COMB (L/100 km)", axis=1)
y = data["COMB (L/100 km)"]

# Teile den Datensatz in Trainings- und temporäre Sets auf (85% Training + Validierung, 15% Test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

# Teile das temporäre Set erneut in Trainings- und Validierungssets auf (70% / 15% der Gesamtmenge)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15 / 0.85), random_state=1)

mlflow.set_experiment("random_forest_regressor_hyperparameter_tuning")

n_estimators = [10, 50, 100, 200]
max_depth = [None, 10, 20, 30]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_features = [1.0, 'sqrt']

def train_rf_model(params):
    with mlflow.start_run():
        # Erstellen des Modells mit den angegebenen Hyperparametern
        rf = RandomForestRegressor(n_estimators=params['n_estimators'],
                                   max_depth=params['max_depth'],
                                   min_samples_split=params['min_samples_split'],
                                   min_samples_leaf=params['min_samples_leaf'],
                                   max_features=params['max_features'],
                                   random_state=1)
        
        # Trainieren des Modells
        rf.fit(X_train, y_train)
        
        # Vorhersagen auf dem Validierungsset
        y_pred = rf.predict(X_val)
        
        # Berechnen des mittleren quadratischen Fehlers
        mse = mean_squared_error(y_val, y_pred)
        
        # Loggen der Hyperparameter und des mittleren quadratischen Fehlers
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        
        # Loggen des Modells mit bester Parameterkombination (zuvor beste Parameterkombination bestimmt)
        if (params['n_estimators'] == 200 and params['max_depth'] == 20 and params['min_samples_split'] == 5 and params['min_samples_leaf'] == 1 and params['max_features'] == 1.0):
            mlflow.sklearn.log_model(rf, "random_forest_model")
        
        return mse

best_mse = float('inf')
best_params = None

for n in n_estimators:
    for d in max_depth:
        for s in min_samples_split:
            for l in min_samples_leaf:
                for f in max_features:
                    params = {
                        'n_estimators': n,
                        'max_depth': d,
                        'min_samples_split': s,
                        'min_samples_leaf': l,
                        'max_features': f
                    }
                    
                    mse = train_rf_model(params)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = params

print("Best Valid MSE:", best_mse)
print("Best parameters:", best_params)

# Erstellen des Modells mit den besten Hyperparametern
best_rf = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'],
                                max_features=best_params['max_features'],
                                random_state=1)

# Trainieren des Modells mit den Trainings- und Validierungsdaten
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)
best_rf.fit(X_train_val, y_train_val)

# Vorhersagen auf dem Testdatensatz
y_test_pred = best_rf.predict(X_test)

# Berechnen des mittleren quadratischen Fehlers auf dem Testdatensatz
test_mse = mean_squared_error(y_test, y_test_pred)

print("Test MSE:", test_mse)