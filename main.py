import hydra
from hydra import utils
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset.fashion_mnist import create_dataloaders
from loss.loss import get_loss_fn
from model.sample import SampleModel
from optimizer.optimizer import get_optimizer
from utils import get_device, check_dir


class Trainer:
    def __init__(
        self,
        device: str = "gpu",
        seed: int = 42,
        max_epoch: int = 1,
        train_bs: int = 1,
        valid_bs: int = 1,
        test_bs: int = 1,
        lr: float = 1e-2,
        accumulation_step: int = 1,
        log_step: int = 100,
        is_print_config: bool = True,
        weight_filename: str = "sample",
        wandb_api_key: str = "",
        wandb_project: str = "project",
        wandb_name: str = "display-name",
    ):
        self.device = get_device(device)
        self.fix_seed(seed)
        self.seed = seed
        self.max_epoch = max_epoch
        self.train_batch_size = train_bs
        self.valid_batch_size = valid_bs
        self.test_batch_size = test_bs
        self.lr = lr
        self.accumulation_step = accumulation_step
        self.log_step = log_step
        self.wandb_api_key = wandb_api_key
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name

        self.init_variables()
        self.set_paths(weight_filename)
        if is_print_config:
            self.print_config()

    def fix_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_variables(self):
        self.dataloader_pbar = None
        self.epoch = 0
        self.log_dict = dict()
        self.loss_fn = None
        self.model = None
        self.optimizer = None

    def set_paths(self, weight_filename):
        project_dir = utils.get_original_cwd()

        self.project_dir = Path(project_dir).absolute()
        self.data_dir = check_dir(self.project_dir / "data")
        self.weight_dir = check_dir(self.project_dir / "weight")
        self.model_save_path = self.weight_dir / f"{weight_filename}.pth"

    def print_config(self):
        print("[ Config ]")

        print("\n- Path -")
        print(f"Project dir: {self.project_dir}")
        print(f"Data dir: {self.data_dir}")
        print(f"Weight dir: {self.weight_dir}")
        print(f"Model save path: {self.model_save_path}")

        print("\n- Trainer -")
        print(f"Device : {self.device}")
        print(f"Seed : {self.seed}")
        print(f"Max epoch: {self.max_epoch}")
        print(f"Gradient Accumulation : {self.accumulation_step}")
        print(f"Batch size - Train: {self.train_batch_size}")
        print(f"           - Valid: {self.valid_batch_size}")
        print(f"           - Test : {self.test_batch_size}")
        print(
            f"           - Train (Accumulated): {self.train_batch_size * self.accumulation_step}"
        )
        print(f"Learning rate : {self.lr}")

        print("\n- W&B -")
        print(f"API Key : {self.wandb_api_key[:10]}...")
        print(f"Project : {self.wandb_project}")
        print(f"Name : {self.wandb_name}")

        print(f"\n{'-'*30}\n")

    def init_wandb(self):
        os.environ["WANDB_API_KEY"] = self.wandb_api_key
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_name,
            config={
                "device": self.device,
                "seed": self.seed,
                "max_epoch": self.max_epoch,
                "batch_size_train": self.train_batch_size,
                "batch_size_valid": self.valid_batch_size,
                "batch_size_test": self.test_batch_size,
                "learning_rate": self.lr,
            },
        )
        wandb.watch(self.model, log_freq=self.log_step)

    def set_dataloader_pbar(
        self,
        dataloader: DataLoader,
        desc: str,
        leave: bool = False,
    ):
        self.dataloader_pbar = tqdm(dataloader, leave=leave)
        self.dataloader_pbar.set_description(desc)
        self.dataloader_pbar.set_postfix(self.log_dict)

    def train(self, dataloader: DataLoader):
        self.model.train()
        self.set_dataloader_pbar(dataloader, desc=f"Epoch {self.epoch}")

        for step, (X, y) in enumerate(self.dataloader_pbar, start=1):
            X, y = X.to(self.device), y.to(self.device)

            # Forward
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backward
            loss.backward()
            if step % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            if step % self.log_step == 0:
                self.log_dict["loss"] = loss.item()
                self.dataloader_pbar.set_postfix(self.log_dict)
                self.global_step = self.epoch * len(dataloader) + step
                wandb.log(
                    {"train/loss": self.log_dict["loss"], "epoch": self.epoch},
                    step=self.global_step,
                )

    def validation(self, dataloader: DataLoader):
        self.model.eval()
        self.set_dataloader_pbar(dataloader, desc=f"Valid {self.epoch}")

        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for step, (X, y) in enumerate(self.dataloader_pbar, start=1):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y).item()
                total_loss += loss
                correct += (pred.argmax(1) == y).sum().item()

                # Logging for step
                if step % self.log_step == 0:
                    self.log_dict["loss"] = loss
                    self.dataloader_pbar.set_postfix(self.log_dict)

        # Logging
        total_loss /= len(dataloader)
        acc = correct * 100 / len(dataloader.dataset)
        self.log_dict["val_loss"] = total_loss
        self.log_dict["val_acc"] = f"{(acc):>0.1f}%"
        wandb.log({"valid/loss": total_loss}, step=self.global_step)
        wandb.log({"valid/accuracy": acc}, step=self.global_step)

    def test(self, dataloader: DataLoader):
        self.model.eval()
        self.set_dataloader_pbar(dataloader, desc="Test", leave=True)

        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for step, (X, y) in enumerate(self.dataloader_pbar, start=1):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y).item()
                total_loss += loss
                correct += (pred.argmax(1) == y).sum().item()

                # Logging for step
                if step % self.log_step == 0:
                    self.log_dict["loss"] = loss
                    self.dataloader_pbar.set_postfix(self.log_dict)

        # Logging
        total_loss /= len(dataloader)
        acc = correct * 100 / len(dataloader.dataset)
        print(f"Test loss: {total_loss}")
        print(f"Test accuracy: {(acc):>0.1f}%")
        wandb.log({"test/loss": total_loss})
        wandb.log({"test/accuracy": acc})

    def save_weight(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Saved model state: {self.model_save_path}")

    def run(self):
        train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
            data_dir=self.data_dir,
            train_batch_size=self.train_batch_size,
            valid_batch_size=self.valid_batch_size,
            test_batch_size=self.test_batch_size,
        )
        self.model = SampleModel().to(self.device)
        self.loss_fn = get_loss_fn()
        self.optimizer = get_optimizer(self.model, self.lr)

        self.init_wandb()

        try:
            for self.epoch in range(self.max_epoch):
                self.train(train_dataloader)
                self.validation(valid_dataloader)
        except KeyboardInterrupt:
            print("Stop training")

        self.save_weight()
        self.test(test_dataloader)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(
        device=cfg.device,
        seed=cfg.seed,
        max_epoch=cfg.max_epoch,
        train_bs=cfg.train_bs,
        valid_bs=cfg.valid_bs,
        test_bs=cfg.test_bs,
        lr=cfg.lr,
        accumulation_step=cfg.accumulation_step,
        log_step=cfg.log_step,
        is_print_config=cfg.is_print_config,
        weight_filename=cfg.weight_filename,
        wandb_api_key=cfg.wandb_api_key,
        wandb_project=cfg.wandb_project,
        wandb_name=f"{cfg.wandb_name}-{cfg.now}",
    )
    trainer.run()


if __name__ == "__main__":
    main()
