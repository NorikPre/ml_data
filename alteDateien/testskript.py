import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import re

# Lade die .csv-Datei als DataFrame
data = pd.read_csv("autos.csv")

# Liste der Spalten, die kodiert werden sollen
columns_to_encode = ['MAKE', 'VEHICLE CLASS', 'TRANSMISSION']

# Target Encoder erstellen
encoder = ce.TargetEncoder(cols=columns_to_encode)

# Encoder an die Daten anpassen
encoder.fit(data[columns_to_encode], data['FUEL CONSUMPTION'])

# Kodierung anwenden
encoded_data = encoder.transform(data[columns_to_encode])

# Namen der neu erstellten Spalten ändern
encoded_data.columns = [f"{col}_enc" for col in columns_to_encode]

# Kodierte Daten mit dem ursprünglichen DataFrame zusammenführen
data = pd.concat([encoded_data, data], axis=1)


# Umcodierung der Werte in der Spalte FUEL
data["FUEL"] = data["FUEL"].replace({'D': 1, 'X': 2, 'Z': 3, 'N': 4, 'E': 5})
# Entferne die aufgelisteten Spalten aus dem DataFrame
columns_to_remove = ["MODEL",'MAKE', 'VEHICLE CLASS', "HWY (L/100 km)", "COMB (L/100 km)", "COMB (mpg)", "EMISSIONS"]
data = data.drop(columns_to_remove, axis=1)


# Beispiel: Erstelle ein neues Feature namens "ENGINE_CYLINDER_RATIO" als Verhältnis von ENGINE SIZE zu CYLINDERS
data["ENGINE_CYLINDER_RATIO"] = data["ENGINE SIZE"] / data["CYLINDERS"]

data["TRANSMISSION_TYPE"] = data["TRANSMISSION"].apply(lambda x: 0 if "A" in x else 1)
# Entferne die Spalte "TRANSMISSION"
data = data.drop("TRANSMISSION", axis=1)


# Trenne die Zielvariable von den Features
X = data.drop("FUEL CONSUMPTION", axis=1)
y = data["FUEL CONSUMPTION"]

# Teile den Datensatz in Trainings- und temporäre Sets auf (85% Training + Validierung, 15% Test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Teile das temporäre Set erneut in Trainings- und Validierungssets auf (70% / 15% der Gesamtmenge)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42)