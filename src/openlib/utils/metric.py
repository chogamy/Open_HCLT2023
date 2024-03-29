"""
    sinlge
     - acc
     - f1-score
     - ind-f1-score
     - ood-f1-score
    multi
     - binary
        - auroc
        - fpr95
        - aupr in
        - aupr out
     - base
        - ind, ood, mix 구별에 대한 acc, f1-score
        - ind, mix 에 대해서 세부 라벨 예측에 대한 acc, f1-score
    visulization
    : confustionmatrix
"""
import torch
from torch import nn
import torchmetrics
import lightning.pytorch as pl
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

class Metric(pl.LightningModule):
    def __init__(self, num_classes, num_labels=None, pre_train=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.pre_train = pre_train
        self.results = {}
        
        if self.num_labels is None:
            self.metrics = nn.ModuleDict({
                'ACC': torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average=None),
                'f1': torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average=None)
            })
            
        if self.num_labels:
            self.metrics = nn.ModuleDict({
                'ACC': torchmetrics.Accuracy(task="multilabel", num_labels=self.num_classes, average=None),
                'f1': torchmetrics.F1Score(task="multilabel", num_labels=self.num_classes, average=None),
                'micro-f1': torchmetrics.F1Score(task="multilabel", num_labels=self.num_classes, average="micro"),
                'weighted-f1': torchmetrics.F1Score(task="multilabel", num_labels=self.num_classes, average="weighted"),
                'em': torchmetrics.ExactMatch(task='multiclass', num_classes=2, multidim_average='global'),
                'hamming': torchmetrics.HammingDistance(task="multiclass", num_classes=2), 
                "auroc": torchmetrics.AUROC(task='multilabel',num_labels=self.num_classes, average="micro"),
                'n_ints_acc': torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
            })
            
            
    def all_compute(
        self,
        preds,
        labels,
        n_ints_preds=None,
        n_ints_labels=None,
        pre_train=False,
        multi=None,
        test=False,
        binary=None,
        num=None,
        visualization=True,
        
    ):
        p, l = preds, labels
        for key in self.metrics.keys():
            if key == 'n_ints_acc':
                p, l = n_ints_preds, n_ints_labels
        
            if key in ["ACC", 'f1']:
                score = self.metrics[key](p.float(), l.long())
                
                self.results[f"{key}"] = score.mean()
                self.results[f"{key}_ind"] = score[:-1].mean()
                self.results[f"{key}_ood"] = score[-1]
            else:    
                self.results[key] = self.metrics[key](p.float(), l.long())
            
        return self.results

    def all_reset(self):
        for key in self.metrics.keys():
            self.metrics[key].reset()
            
    def all_compute_end(self, pre_train=False, test=False):
        for key in self.metrics.keys():
            if key == 'ACC' or key == 'f1':
                score = self.metrics[key].compute()
                self.results[f"{key}"] = score.mean()
                self.results[f"{key}_ind"] = score[:-1].mean()
                self.results[f"{key}_ood"] = score[-1]
            else:    
                self.results[key] = self.metrics[key].compute()
        
        return self.results

    def end(self, name):
        try:
            df = pd.read_csv(f"./results/{name[:-2]}.csv")
        except FileNotFoundError:
            df = pd.DataFrame()
            
        data = {"model_name": name}
        for k, v in self.results.items():
            data[k] = v.item()

        df = pd.concat([df, pd.DataFrame.from_dict([data])], ignore_index=True)

        df.to_csv(f"./results/{name[:-2]}.csv", index=False)
        # self.save_dict_to_csv(self.results, "result.csv")

    def save_dict_to_csv(self, data_dict, file_path):
        with open(file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
            if csvfile.tell() == 0:
                writer.writeheader()
            # 숫자로 변환하여 저장
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    value = str(value.item())  # tensor를 숫자로 변환
                writer.writerow({key: value})

    def draw_label(self, weights, preds, tsne_path):
        print("TSNE: fitting start...")
        tsne = TSNE(n_components=2, metric='cosine', random_state=0, n_jobs=4)
        embedding = tsne.fit_transform(weights)

        df = pd.DataFrame(embedding, columns=['x', 'y'])  # x, y 값을 DataFrame으로 변환
        df['predict'] = preds  # 라벨을 DataFrame에 추가
        # df.to_csv(self.tsne_path + '.csv', index=False)
        df.to_csv(tsne_path + '.csv', index=False)
    
    def confusion(self, y_true, y_pred, labels, tsne_path):
        # confmat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        cm = confusion_matrix(y_true, y_pred)
        # cm = confmat(y_pred, y_true)

        
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            xlabel='Predicted label',
            ylabel='True label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
        fig.tight_layout()
        plt.savefig(tsne_path + '.pdf')
        plt.close(fig)
    
    def save_predict(self,y_true, y_pred, label_to_id, total_y, tsne_path):
        # personalize 마지막 예측라벨 저장위해서
        label_mapping = {value: key for key, value in label_to_id.items()}
        labels = list(label_mapping.values())
        labels.append('open')
        self.confusion(y_true,y_pred,labels, tsne_path)

        scalar_label = labels[int(total_y[-1].item())]
        print("prediction", scalar_label)

        with open('/workspace/openlib/personalize/output.txt', 'w') as file:
            file.write(str(scalar_label))



if __name__ == "__main__":
    num_classes = 4
    metric = Metric(num_classes, multi=True)