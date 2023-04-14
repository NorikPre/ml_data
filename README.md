# Projekt: Vorhersage des Kraftstoffverbrauchs mit Random Forest Regressor

## Einleitung

In diesem Projekt verwenden wir einen Datensatz mit dem Namen "Fuel Consumption 2000-2022", um ein maschinelles Lernmodell zu erstellen, das den Kraftstoffverbrauch von Fahrzeugen vorhersagen kann. Unser Modell basiert auf dem Random Forest Regressor Algorithmus, der ein leistungsfähiger und weit verbreiteter Algorithmus zur Lösung von Regressionsproblemen ist. Das Hauptziel dieses Projekts ist es, ein zuverlässiges und genaues Modell zu erstellen, das den kombinierten (innerorts, außerorts und Autobahn) Kraftstoffverbrauch auf der Grundlage verschiedener Merkmale von Fahrzeugen, wie z.B. Jahr, Marke, Modell, Fahrzeugklasse, Motorgröße, Zylinderzahl, Getriebeart und Treibstoffart, vorhersagen kann.

## Kapitel 1: Auswahl Datensatz von Kaggle, Erläuterung und Vorstellung in der README

### Auswahl des Datensatzes

Der Datensatz "Fuel Consumption 2000-2022" wurde von der Plattform Kaggle ausgewählt. 

### Erläuterung und Vorstellung des Datensatzes

Der "Fuel Consumption 2000-2022" Datensatz enthält Informationen über den Kraftstoffverbrauch von Fahrzeugen aus den Jahren 2000 bis 2022. Jede Zeile im Datensatz repräsentiert ein Fahrzeugmodell und enthält die folgenden Spalten (gelb die Spalten, mit denen ich arbeiten werde):

- `YEAR`: das Baujahr des Fahrzeugs
- `MAKE`: die Marke des Fahrzeugs
- `MODEL`: der Modellname des Fahrzeugs
- `VEHICLE CLASS`: die Fahrzeugklasse (z.B. SUV, Two-Seater)
- `ENGINE SIZE`: die Größe des Motors in Litern
- `CYLINDERS`: die Anzahl der Zylinder
- `TRANSMISSION`: Art des Getriebes (z.B. Automatik oder Schaltgetriebe) und Anzahl der Gänge 
- `FUEL`: Art des verwendeten Kraftstoffs (z.B. (Premium-)Benzin, Diesel)
- FUEL CONSUMPTION: Kraftstoffverbrauch in Litern pro 100 Kilometer (L/100 km)
- HWY (L/100 km): Kraftstoffverbrauch auf der Autobahn in Litern pro 100 Kilometer (L/100 km)
- `COMB (L/100 km)`: Kombinierter Kraftstoffverbrauch in Litern pro 100 Kilometer (L/100 km)
- COMB (mpg): Kombinierter Kraftstoffverbrauch in Meilen pro Gallone (mpg)
- EMISSIONS: CO2-Emissionen in Gramm pro Kilometer (g/km)

In diesem Projekt konzentrieren wir uns auf die Vorhersage der Spalte "`COMB (L/100 km)`" auf der Grundlage der verfügbaren Merkmale im Datensatz.


## Auswahl Datensatz von Kaggle
## Datenvorbearbeitung im Code
## Feature Engineering
## Split des Datensatzes
## Auswahl ML Methode
## Auswahl geeigneter Metriken
## Training ML Algorythmus
## Tuning der Hyperparameter mit MLFlow
## Evaluation auf Testdaten
## Erläuterung möglicher Schwachstellen und Verbesserungsmöglichkeiten
