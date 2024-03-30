#!/bin/bash

export TOKENIZERS_PARALLELISM=false

datas=(mixsnips_clean mixatis_clean)
plms=(bert-base-uncased roberta-base google/electra-base-discriminator distilbert-base-uncased)
seeds=(0 1 2 3 4 5 6 7 8 9) 
ratios=(0.25 0.5 0.75) 


for data in ${datas[@]}
do
    for plm in ${plms[@]}
    do
        for ratio in ${ratios[@]}
        do
            for seed in ${seeds[@]}
            do
            python main.py \
            --model_name_or_path ${plm} \
            --known_cls_ratio ${ratio} \
            --model samples/model/ours.yaml --data samples/data/${data}.yaml \
            --trainer samples/trainer/ours.yaml \
            --seed ${seed} --mode train
            python main.py \
            --config ./trainer_logs/Ours_${plm}_${data}${ratio}_${seed}.yaml --mode test
            done
        done
    done
done
