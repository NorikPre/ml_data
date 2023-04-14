import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Daten laden
data = pd.read_csv("daten.csv")

# Zielvariable und Merkmale trennen
X = data.drop(["FUEL CONSUMPTION", "ENGINE_CYLINDER_RATIO", "TRANSMISSION_TYPE"], axis=1)
y = data["FUEL CONSUMPTION"]

# Daten aufteilen: 70% Trainingsdaten, 15% Validierungsdaten, 15% Testdaten
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Parameter-Raum definieren
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", "log2"],
}

# GridSearchCV mit RandomForestRegressor erstellen
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# GridSearchCV auf Trainingsdaten anwenden
grid_search.fit(X_train, y_train)

# Beste Parameter ausgeben
print("Beste Parameter:", grid_search.best_params_)

# Vorhersage auf Validierungsdaten
y_val_pred = grid_search.predict(X_val)

# Validierungsfehler (MSE) berechnen
val_error = mean_squared_error(y_val, y_val_pred)
print("Validierungsfehler (MSE):", val_error)

# Vorhersage auf Testdaten
y_test_pred = grid_search.predict(X_test)

# Testfehler (MSE) berechnen
test_error = mean_squared_error(y_test, y_test_pred)
print("Testfehler (MSE):", test_error)
