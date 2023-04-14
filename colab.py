from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt

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
encoder.fit(data[columns_to_encode], data['EMISSIONS'])

# Kodierung anwenden
encoded_data = encoder.transform(data[columns_to_encode])

# Namen der neu erstellten Spalten ändern
encoded_data.columns = [f"{col}_enc" for col in columns_to_encode]

# Kodierte Daten mit dem ursprünglichen DataFrame zusammenführen
data = pd.concat([encoded_data, data], axis=1)


# Umcodierung der Werte in der Spalte FUEL
data["FUEL"] = data["FUEL"].replace({'D': 1, 'X': 2, 'Z': 3, 'N': 4, 'E': 5})
# Entferne die aufgelisteten Spalten aus dem DataFrame
columns_to_remove = ["MODEL",'MAKE', 'VEHICLE CLASS', "FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (mpg)", "COMB (L/100 km)"]
data = data.drop(columns_to_remove, axis=1)


# Beispiel: Erstelle ein neues Feature namens "ENGINE_CYLINDER_RATIO" als Verhältnis von ENGINE SIZE zu CYLINDERS
data["ENGINE_CYLINDER_RATIO"] = data["ENGINE SIZE"] / data["CYLINDERS"]

data["TRANSMISSION_TYPE"] = data["TRANSMISSION"].apply(lambda x: 0 if "A" in x else 1)
# Entferne die Spalte "TRANSMISSION"
data = data.drop("TRANSMISSION", axis=1)

data = pd.read_csv("autos.csv")

abc = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(abc, annot=True, cmap='coolwarm', fmt='.2f')

plt.show()