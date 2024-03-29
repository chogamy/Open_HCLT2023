# Open Intent Classification (WIP)


## Key features
- [**Transformers**](https://https://huggingface.co/docs/transformers/index) 
- [**Lightning**](https://lightning.ai//) 


## 실행 방법1
config.yaml에 모델, 데이터, Trainer를 지정
```bash
python train.py --config <config.yaml>
```

실행 예제
```bash 
python train.py --config samples/feature_extractor.yaml
```

## 실행 방법2
모델, 데이터, Trainer를 각각 따로 지정
```bash 
python train.py --model <model-yaml> --trainer <trainer-yaml> --data <data-yaml> --model_name_or_path <plm-path> --known_cls_ratio <float> --seed <int> --mode <train-or-test>
```

실행 예제
```bash 
python train.py --model samples/model/adb.yaml --trainer samples/trainer/adb.yaml --data samples/data/stackvoerflow.yaml --model_name_or_path bert-base-cased --known_cls_ratio 0.25 --seed 5 --mode train
```

