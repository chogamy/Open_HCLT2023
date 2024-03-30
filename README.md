# [**의도 정보를 활용한 다중 레이블 오픈 의도 분류**](https://koreascience.kr/article/CFKO202306643317218.page)

## Key features
- [**Transformers**](https://https://huggingface.co/docs/transformers/index) 
- [**Lightning**](https://lightning.ai//) 


## 준비
```
pip install -r requirements.txt
```

## 실험 진행

제안 모델
```
bash ours.sh
```

베이스 모델
```
bash naive.sh
```

## 실험 결과

MixATIS

|Known class ratio|모델|Accuracy|
|25|baseline||
|25|our|76.15|
|50|baseline||
|50|our|91.41|
|75|baseline||
|75|our|94.49|

MixSNIPS

|Known class ratio|모델|Accuracy|
|25|baseline|36.01|
|25|our|57.91|
|50|baseline||
|50|our|82.22|
|75|baseline||
|75|our|86.92|




## 실행 방법1
config.yaml에 모델, 데이터, Trainer를 지정
```bash
python train.py --config <config.yaml>
```

실행 예제
```bash 
python train.py --config trainer_logs/Naive_bert-base-uncased_mixsnips_clean0.25_0.yaml
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