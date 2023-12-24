from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

from experiments.data.mnist import MNISTDataModule
from experiments.mnist_partial_radial import PartialRadialLayerMNISTClassifier


def ray_experiment_from_config(config: dict):
    data = MNISTDataModule(data_dir='/tmp/mnist', batch_size=config.get("batch_size", 64))
    model = PartialRadialLayerMNISTClassifier(**config)

    early_stopping = EarlyStopping('val/hard_loss', mode='min', patience=10)

    trainer = Trainer(
        devices="auto",
        accelerator="cpu",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback(), early_stopping],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, data)


if __name__ == "__main__":
    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=False, resources_per_worker={"CPU": 3}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=4,
            checkpoint_score_attribute="val/accuracy",
            checkpoint_score_order="max",
        ),
        callbacks=[WandbLoggerCallback(project="RadialLayer", group="Ray Train")]
    )

    ray_trainer = TorchTrainer(
        ray_experiment_from_config,
        train_loop_config={
            "learning_rate": 1e-3,
            "batch_size": 64,
            "phase_change_epoch": 3,
            "layer1_depth": 3,
            "layer2_depth": 3,
            "spread_lambda": 1.0,
            "quantile_lambda": 1.0,
            "load_balancing_lambda": 1.0,
            "quantile_history_weight": 0.3,
            "max_power": 4
        },
        scaling_config=scaling_config,
        run_config=run_config,
    )
    ray_trainer.fit()
