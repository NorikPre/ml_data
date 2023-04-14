# Ca 0.4648
#0.4392686057265698

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Daten laden
data = pd.read_csv("autos.csv")

# Zielvariable und Merkmale trennen
X = data.drop(["FUEL CONSUMPTION", "HWY (L/100 km)","COMB (L/100 km)","COMB (mpg)","EMISSIONS"], axis=1)
y = data["FUEL CONSUMPTION"]

# Kategorische Spalten in numerische Werte umwandeln
X = pd.get_dummies(X, columns=["MAKE", "MODEL", "VEHICLE CLASS", "TRANSMISSION", "FUEL"], drop_first=True)


# Daten aufteilen: 80% Trainingsdaten, 20% Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Random Forest Regressor erstellen
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, max_features=1.0, random_state=None)

# Modell auf Trainingsdaten trainieren
rf.fit(X_train, y_train)

# Vorhersage auf Testdaten
y_test_pred = rf.predict(X_test)

# Testfehler (MSE) berechnen
test_error = mean_squared_error(y_test, y_test_pred)
print("Testfehler (MSE):", test_error)

