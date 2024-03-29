import random

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)


class TransformerFeatureExtractor(nn.Module):
    """Extract feature using Transformer

    Examples:
        With custom models:

            >>> from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
            >>> feature_extractor = TransformerFeatureExtractor(
                    model_name_or_path="path.to.checkpoints",
                )
            >>> features = feature_extractor(input)
    """

    def __init__(self, model_name_or_path: str, dropout_prob: float = 0.5):
        super().__init__()

        # Use pretrained language model
        self.model, self.dense = self.initialize_feature_extractor(model_name_or_path)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, batch, pooling=True):
        outputs = self.model(**batch)

        if pooling==False:
            return outputs  # last_hidden_state, pooler_output

        mean_pooling = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dense(mean_pooling)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        return pooled_output

    def initialize_feature_extractor(self, model_name_or_path: str):
        if model_name_or_path.endswith("ckpt"):
            # TODO: should be fixed!
            print("CKPT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            pretrained = torch.load(model_name_or_path)
            model_name_or_path = pretrained["hyper_parameters"]["model_name_or_path"]
            state_dict = pretrained["state_dict"]
            for key in list(state_dict.keys()):
                state_dict[key.replace("model.model.", "")] = state_dict.pop(key)
            # print("dense", state_dict.keys())

            dense_w = state_dict.pop("model.dense.weight")
            dense_b = state_dict.pop("model.dense.bias")

            model = AutoModel.from_pretrained(model_name_or_path, state_dict=state_dict)

            dense = nn.Linear(model.config.hidden_size, model.config.hidden_size)

            dense.weight.data = dense_w
            dense.bias.data = dense_b

        else:
            print("NOT CKPT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = AutoModel.from_pretrained(model_name_or_path, config=config)
            dense = nn.Linear(model.config.hidden_size, model.config.hidden_size)

        return model, dense


class MixText(TransformerFeatureExtractor):
    """Only forward loop is different from BERT"""

    def __init__(
        self, model_name_or_path: str, dropout_prob: float = 0.5, n: int = 1, l: float = 0.3
    ):
        super().__init__(model_name_or_path, dropout_prob)

        self.n = n

    def forward(self, batch0, batch1=None):
        if batch1 is None:
            """
            This means pre-training or inference
            """

            outputs = self.model(**batch0)

            mean_pooling = outputs.last_hidden_state.mean(dim=1)
            pooled_output = self.dense(mean_pooling)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            return outputs.last_hidden_state, pooled_output

        """
        Mix up finetuning
        """

        emb = self.model.embeddings
        layers = self.model.encoder.layer

        hidden0 = emb(batch0["input_ids"])
        hidden1 = emb(batch1["input_ids"])

        attention_mask0 = batch0["attention_mask"]
        input_shape0 = batch0["input_ids"].size()

        attention_mask1 = batch1["attention_mask"]
        input_shape1 = batch1["input_ids"].size()

        attention_mask0: torch.Tensor = self.model.get_extended_attention_mask(
            attention_mask0, input_shape0
        )
        attention_mask1: torch.Tensor = self.model.get_extended_attention_mask(
            attention_mask1, input_shape1
        )

        for i, layer in enumerate(layers[: self.n]):
            layer_outputs0 = layer(hidden0, attention_mask0)
            layer_outputs1 = layer(hidden1, attention_mask1)

            hidden0 = layer_outputs0[0]
            hidden1 = layer_outputs1[0]

        l = random.random()
        hidden = l * hidden0 + (1.0 - l) * hidden1

        # assert 1==0, "어텐션 마스크 0이야 1이야?"

        for i, layer in enumerate(layers[self.n :]):
            layer_outputs = layer(hidden, attention_mask0)
            hidden = layer_outputs[0]

        mean_pooling = hidden.mean(dim=1)
        pooled_output = self.dense(mean_pooling)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        return hidden, pooled_output
