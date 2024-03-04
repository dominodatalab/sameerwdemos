import mlflow
import torch
import ray
import os
import argparse
import ray.util
import pytorch_lightning as pl
from ray import tune
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from ray import air, tune
from ray.train import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.lightning import (
    RayDDPStrategy,
    RayTrainReportCallback,
)

print("Initializing Ray Cluster...")
service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
ray.init(f"ray://{service_host}:{service_port}")

class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)
        self.eval_loss = []
        self.eval_accuracy = []

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_acc)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        with FileLock(os.path.expanduser("~/.data.lock")):
            return MNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)
        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size), num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size), num_workers=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def train_mnist_tune(config, num_epochs=5, data_dir="~/data"):
    data_dir = os.path.expanduser(data_dir)
    model = LightningMNISTClassifier(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        enable_progress_bar=False,
        callbacks=[RayTrainReportCallback()]
    )
    trainer.fit(model)
    mlflow.pytorch.log_model(model, "model")

def tune_mnist(storage_path, num_epochs, num_trials, cpus_per_trial, gpus_per_trial, data_dir="~/data"):
    config = {
        "layer_1_size": tune.choice([16, 32, 64]),
        "layer_2_size": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    train_fn_with_parameters = tune.with_parameters(train_mnist_tune,num_epochs=num_epochs,data_dir=data_dir)
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_trials,
        ),
        run_config=air.RunConfig(
            name="tune_mnist_asha",
            storage_path=storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute="val_loss", checkpoint_score_order="min"),
            callbacks=[MLflowLoggerCallback(experiment_name="mnist-tuning", save_artifact=True)]
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

parser = argparse.ArgumentParser(description="Hyperparameter tuning an MNIST model using Ray and Pytorch Lightning")
parser.add_argument("--storage_path", required=False, default="/mnt/data/ray-results", type=str)
parser.add_argument("--num_epochs", required=False, default=5, type=int)
parser.add_argument("--num_trials", required=False, default=4, type=int)
parser.add_argument("--cpus_per_trial", required=False, default=2, type=int)
parser.add_argument("--gpus_per_trial", required=False, default=1, type=int)
args = parser.parse_args()

tune_mnist(storage_path=args.storage_path, num_epochs=args.num_epochs, num_trials=args.num_trials, cpus_per_trial=args.cpus_per_trial, gpus_per_trial=args.gpus_per_trial)
