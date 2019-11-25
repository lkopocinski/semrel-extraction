import mlflow

mlflow.set_tracking_uri("http://10.17.50.132:8080")
print(mlflow.get_tracking_uri())

mlflow.set_experiment("no_exp")

mlflow.log_metric("metric", 42)
for i in range(1, 100):
    mlflow.log_metric(key="loss", step=i, value=i*2)

print(mlflow.get_artifact_uri())
mlflow.log_artifact("./test.txt")
