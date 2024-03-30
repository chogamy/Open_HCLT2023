"""
기존 aik에 리니어 레이어 하나 쌓은거
그대로 training 해보고
그 다음 known intents를 예측해보자
"""

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler

from src.openlib.models import BASE
from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
from src.openlib.utils.metric import Metric


class Naive(BASE):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-3,
        plm_lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_rate: int = 0,
        freeze=None,
        method: str = "aik",  # aik, bce
        sampler: str = None,
        sampler_config: dict = None,
    ):
        if sampler:
            sampler_config["unseen_label_id"] = num_classes
        super().__init__(sampler=sampler, sampler_config=sampler_config)
        self.save_hyperparameters()

        self.method = method
        self.num_classes = num_classes  # unknwon은 포함 안함

        self.model = TransformerFeatureExtractor(
            model_name_or_path, dropout_prob=dropout_prob
        )
        self.softmax = nn.Softmax(dim=1)

        # 내가 추가한 부분~~~~ ###################
        # self.classifier = nn.Linear(self.model.dense.out_features, num_classes)
        self.classifier = nn.Linear(self.model.dense.out_features, num_classes + 1)

        self.sigmoid = nn.Sigmoid()

        self.metric = Metric(
            self.num_classes + 1, num_labels=self.num_classes + 1
        )  # unk 포함

    def forward(self, batch):
        if "outputs" in batch:
            outputs = batch["outputs"]
        else:
            outputs = self.model(batch, pooling=False)
            outputs = outputs.last_hidden_state

        # b i
        i_logits = self.classifier(outputs.mean(dim=1))

        return i_logits
        # return i_logits, n_ints, rep, weight.transpose(1,2)

    def training_step(self, batch, batch_idx):
        model_input, labels, n_ints_labels = self.step(batch, batch_idx)

        i_logits = self(model_input)

        # loss = F.binary_cross_entropy_with_logits(
        #     i_logits, labels[:, : self.num_classes]
        # )

        loss = F.binary_cross_entropy_with_logits(i_logits, labels)

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def prediction(self, i_logits):

        i_probs = self.sigmoid(i_logits)

        threshold = 0.5

        i_probs[i_probs < threshold] = 0
        i_probs[i_probs >= threshold] = 1

        print(i_probs)

        i_pred = i_probs

        """
        [0, 1, 0, 1] -> [2]
        [0, 0, 0, 1] -> [1]
        """
        n_ints_pred = torch.sum(i_pred, dim=1)

        return i_pred, n_ints_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        device = self.device

        model_input, labels, n_ints_labels = self.step(batch, batch_idx)

        i_logits = self(model_input)

        i_preds, n_preds = self.prediction(i_logits)

        metric = self.metric.all_compute(
            i_preds,
            labels,
            n_ints_preds=n_preds,
            n_ints_labels=n_ints_labels,
            pre_train=False,
        )

        self.log_dict({"val_acc": metric["em"]})

    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        self.metric.all_reset()

    def test_step(self, batch, batch_idx):
        self.eval()

        device = self.device

        model_input, labels, n_ints_labels = self.step(batch, batch_idx)

        i_logits = self(model_input)

        i_preds, n_preds = self.prediction(i_logits)

        metric = self.metric.all_compute(
            i_preds,
            labels,
            n_ints_preds=n_preds,
            n_ints_labels=n_ints_labels,
            pre_train=False,
        )

        self.log_dict({"test_acc": metric["em"]})

    def on_test_epoch_end(self):
        self.eval()
        self.log_dict(
            self.metric.all_compute_end(pre_train=False, test=True), prog_bar=True
        )
        self.metric.all_reset()

        self.metric.end(self.trainer.checkpoint_callback.filename)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        plm_param_names = set(f"model.{n}" for n, p in self.model.named_parameters())

        plm_params = (
            (n, p) for n, p in self.named_parameters() if n in plm_param_names
        )
        rest_params = (
            (n, p) for n, p in self.named_parameters() if n not in plm_param_names
        )

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in plm_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.plm_lr,
            },
            {
                "params": [p for n, p in plm_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.plm_lr,
            },
            {
                "params": [
                    p for n, p in rest_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.lr,
            },
            {
                "params": [
                    p for n, p in rest_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.hparams.lr,
            },
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters)

        warm_up_steps = int(
            self.trainer.estimated_stepping_batches * self.hparams.warmup_rate
        )
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            # num_warmup_steps=self.hparams.warmup_steps,
            num_warmup_steps=warm_up_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
