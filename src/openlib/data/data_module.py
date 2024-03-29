import os
import random
from itertools import chain

from torch.utils.data import DataLoader
import datasets
import lightning.pytorch as pl
from datasets import load_dataset, concatenate_datasets, load_from_disk, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer


from transformers import AutoTokenizer, default_data_collator


class OICDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        dataset: str,
        preprocessed_dir: str = None,
        model_name_or_path: str = None,
        known_cls_ratio: float = 0.5,
        labeled_ratio: float = 1.0,
        include_unknown: bool = False,
        max_seq_len: int = 45,
        batch_size: int = 32,
        num_workers: int = 8,
        k_1: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_path = data_path
        # if dataset in ['banking', 'mixsnips_clean', 'oos', 'stackoverflow']:
        #     self.data_path = "nahyeon00/" + self.dataset
        # elif dataset in ['mixatis_clean']:
        #     self.data_path = "gamy0315/" + self.dataset
        # elif dataset in ['kmhas']:
        #     self.data_path = 'jeanlee/kmhas_korean_hate_speech'
        # else:
        #     raise ValueError('Invalid dataset name')

        self.model_name_or_path = model_name_or_path

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.known_cls_ratio = known_cls_ratio
        self.labeled_ratio = labeled_ratio
        self.include_unknown = include_unknown
        self.num_classes = None  # set after preparing
        self.pre_dir = preprocessed_dir
        self.k_1 = k_1

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )

    def prepare_data(self):
        # ? ????????????????????????
        if self.pre_dir and os.path.isdir(self.pre_dir):
            return

        # tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        tokenizer = self.tokenizer

        try:
            raw_datasets = load_dataset(self.data_path)
        except:
            print("@ Warning: dataset is not in hub @@@@@@@@@@@@@@@@@@@@@@@@")
            data_files = {
                "train": self.train_data_path,
                "validation": self.val_data_path,
                "test": self.test_data_path,
            }
            raw_datasets = load_dataset(
                "csv", data_files=self.data_path, delimiter="\t"
            )

        if self.k_1:
            squad_dataset = load_dataset("nahyeon00/SQUAD")
            outlier_dataset_dict = squad_dataset["train"].map(
                lambda example: {"text": example["text"], "label": example["label"]}
            )
            outlier_dataset = DatasetDict({"outlier": outlier_dataset_dict})

        if "mix" in self.dataset:
            mix_intent = [
                intent[0].split("#") for intent in raw_datasets["train"]["intent"]
            ]

            mix_intent = set(chain.from_iterable(mix_intent))
            all_label_list = list(mix_intent)
            all_label_list.sort()
        elif self.dataset == "kmhas":
            all_label_list = [i for i in range(0, 9)]
        else:
            all_label_list = raw_datasets["train"].unique("label")

        n_known_cls = round(len(all_label_list) * self.known_cls_ratio)
        known_label_list = random.sample(all_label_list, k=n_known_cls)

        if self.k_1:
            outlier_label_list = outlier_dataset["outlier"].unique("label")
            known_label_list += outlier_label_list
            raw_datasets["train"] = concatenate_datasets(
                [raw_datasets["train"], outlier_dataset["outlier"]]
            )

        self.num_classes = len(known_label_list)

        label_to_id = {v: i for i, v in enumerate(known_label_list)}
        self.label_to_id = label_to_id

        if "mix" in self.dataset:
            remove_columns = ["token", "tag", "intent"]
        else:
            remove_columns = ["text"]

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=remove_columns,
            fn_kwargs={"label_to_id": label_to_id, "tokenizer": tokenizer},
            load_from_cache_file=False,  # not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # seed value will be added
        self.pre_dir = f"data/preprocessed/{self.dataset}_{self.known_cls_ratio}"
        raw_datasets.save_to_disk(self.pre_dir)

    def setup(self, stage):
        ds = load_from_disk(self.pre_dir)
        if stage in (None, "fit"):
            self.validation = ds["validation"]
            if "mix" in self.dataset:
                ds = ds.filter(lambda x: x["label"][-1] != 1)
            elif self.dataset == "kmhas":
                pass
            else:
                ds = ds.filter(lambda x: x["label"] != -1)
            self.train = ds["train"]

        elif stage == "test":
            self.test = ds["test"]

    def train_dataloader(self):
        loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=default_data_collator,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.validation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_data_collator,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_data_collator,
        )
        return loader

    def preprocess_function(self, examples, label_to_id, tokenizer):
        if "mix" in self.dataset:
            texts = [" ".join(token) for token in examples["token"]]
        else:
            texts = examples["text"]

        result = tokenizer(
            texts, max_length=self.max_seq_len, padding="max_length", truncation=True
        )

        if "mix" in self.dataset:
            result["n_ints_label"] = [
                len(label[0].split("#")) for label in examples["intent"]
            ]
            labels = []
            for intents in examples["intent"]:
                label = [0] * (self.num_classes + 1)
                if self.known_cls_ratio == 1.0:
                    label = [0] * (self.num_classes)
                intents = intents[0].split("#")
                for intent in intents:
                    id = label_to_id.get(intent, self.num_classes)
                    label[id] = 1
                labels.append(label)

            result["label"] = labels

        elif self.dataset == "kmhas":
            result["n_ints_label"] = [len(label) for label in examples["label"]]
            mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            one_hot_labels = mlb.fit_transform(examples["label"])
            result["label"] = one_hot_labels.tolist()

        else:
            result["label"] = [label_to_id.get(l, -1) for l in examples["label"]]

        return result
