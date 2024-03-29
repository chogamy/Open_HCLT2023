from __future__ import annotations
from importlib import import_module

import torch
from omegaconf import DictConfig, ListConfig
import lightning.pytorch as pl

from src.openlib.utils.metric import Metric


def get_model(config: DictConfig | ListConfig):
    module_path, model_name = config.model.class_path.rsplit(".", 1)

    module = import_module(f"src.openlib.models.{module_path}")

    if config.mode == "test":
        ckpt = f"outputs/{config.trainer.callbacks[1].init_args.filename}.ckpt"
        model = getattr(module, f"{model_name}").load_from_checkpoint(ckpt)
    else:
        model = getattr(module, f"{model_name}")(**config.model.init_args)

    return model


class BASE(pl.LightningModule):
    def __init__(
        self,
        # num_classes: int,
        # lr: float = 1e-5,
        # weight_decay: float = 0.01,
        # scheduler_type: str = "linear",
        # warmup_steps: int = 0,
        # freeze=True,
        tsne_path=None,
        tsne_config: dict = None,
        sampler: str = None,
        sampler_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = None  # definition in child class
        self.metric: Metric = None  # definition in child class

        self.tsne_path = None
        if tsne_path:
            self.tsne_path = tsne_path
            self.total_labels = torch.empty(0, dtype=torch.long)
            self.pooled_output = torch.empty((0, self.model.model.config.hidden_size))
            self.total_y_pred = torch.empty(0, dtype=torch.long)
            self.label_to_id = tsne_config["label"]

    def forward(self):
        raise NotImplementedError()

    def step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        if "n_ints_label" in batch:
            model_input = {k: v for k, v in model_input.items() if k != "n_ints_label"}
            return model_input, batch["labels"], batch["n_ints_label"]
        return model_input, batch["labels"], None

    def training_step(self, batch, batch_idx, pooling=True):
        model_input, labels, n_ints_labels = self.step(batch, batch_idx)
        outputs = self.model(model_input, pooling=pooling)

        return outputs, labels, n_ints_labels

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def on_test_epoch_end(self):
        if self.tsne_path:
            ### visualization
            pooled_output = self.pooled_output.cpu().numpy()
            y_true = self.total_labels.cpu().numpy()
            y_pred = self.total_y_pred.cpu().numpy()
            self.metric.draw_label(pooled_output, y_pred, self.tsne_path)  # tsne
            self.metric.save_predict(
                y_true, y_pred, self.label_to_id, self.total_y_pred, self.tsne_path
            )  # confusion matrix, 마지막 라벨 저장

    def visualization(self):
        pass
