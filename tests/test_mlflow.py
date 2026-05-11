import mlflow


def test_mlflow_run_created(tmp_path) -> None:
    tracking_path = tmp_path / "mlruns"
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment("lora-finetune-eval")
    with mlflow.start_run() as run:
        mlflow.log_param("model_name", "gpt2")
        mlflow.log_metric("mock_loss", 0.12)

    client = mlflow.tracking.MlflowClient(mlflow.get_tracking_uri())
    runs = client.search_runs(experiment_ids=[run.info.experiment_id], filter_string="params.model_name = 'gpt2'")
    assert len(runs) == 1
    assert runs[0].data.metrics["mock_loss"] == 0.12
