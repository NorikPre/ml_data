import pandas as pd

# Laden der CSV-Datei in einen Pandas DataFrame
df = pd.read_csv('processed_data.csv')

## Überprüfen, ob es mindestens eine Zeile mit NaN-Werten in Spalte 10 gibt
has_nan = df.iloc[:, 10].isna().any()

if has_nan:
    # Finden der Zeilen, die NaN-Werte in Spalte 10 enthalten und Ausgabe des Index
    nan_rows = df[df.iloc[:, 10].isna()].index
    print("Folgende Zeilen enthalten NaN-Werte in Spalte 10:")
    print(nan_rows)
else:
    print("Es gibt keine NaN-Werte in Spalte 10.")