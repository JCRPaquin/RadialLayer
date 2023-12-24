from lightning.pytorch import Trainer
from ray import tune
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from experiments.data.mnist import MNISTDataModule
from experiments.mnist_multi_partial_radial import MultiPartialRadialLayerMNISTClassifier


def ray_experiment_from_config(config: dict):
    data = MNISTDataModule(data_dir='/tmp/mnist', batch_size=config.get("batch_size", 64))
    model = MultiPartialRadialLayerMNISTClassifier(**config)

    trainer = Trainer(
        devices="auto",
        accelerator="cpu",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, data)


if __name__ == "__main__":
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "phase_change_epoch": tune.choice([-1, 1, 2, 3, 4]),
        "layer1_depth": tune.choice([2, 3, 4]),
        "layer2_depth": tune.choice([2, 3, 4]),
        "spread_lambda": tune.uniform(0.0, 10.0),
        "quantile_lambda": tune.uniform(0.0, 10.0),
        "load_balancing_lambda": tune.uniform(0.0, 10.0),
        "quantile_history_weight": tune.uniform(0.1, 0.6)
    }

    scheduler = ASHAScheduler(
        time_attr="epoch",
        max_t=30,
        grace_period=3,
        reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=False, resources_per_worker={"CPU": 3}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=4,
            checkpoint_score_attribute="val/accuracy",
            checkpoint_score_order="max",
        ),
    )

    ray_trainer = TorchTrainer(
        ray_experiment_from_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val/accuracy",
            search_alg=OptunaSearch(),
            mode="max",
            num_samples=-1,
            scheduler=scheduler,
        ),
    )
    tuner.fit()
