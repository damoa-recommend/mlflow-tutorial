import mlflow

try:
  experiment_id = mlflow.create_experiment("mung")
  experiment = mlflow.get_experiment(experiment_id)

  print("Name: {}".format(experiment.name))
  print("experiment_id: {}".format(experiment.experiment_id))
  print("artifact_location: {}".format(experiment.artifact_location))
  print("tags: {}".format(experiment.tags))
  print("lifecycle_stage: {}".format(experiment.lifecycle_stage))
except Exception as err:
  print('=========== Aready Experiment Message ===========')
  print(err)
  print('=========== Aready Experiment Message ===========')
  print()
  print()