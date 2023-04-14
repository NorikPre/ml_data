import mlflow

experiment_name = "Fuel Consumption Prediction 1"

# Get experiment ID by name
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Delete the experiment by ID
mlflow.delete_experiment(experiment_id)

experiment_name = "Fuel Consumption Prediction 2"

# Get experiment ID by name
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Delete the experiment by ID
mlflow.delete_experiment(experiment_id)