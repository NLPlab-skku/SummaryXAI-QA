# SummaryXAI-NLP
전문지식 대상 판단결과의 이유/근거를 설명가능한 전문가 의사결정 지원 인공지능 기술 개발

설명가능한 오픈도메인 질의응답 시스템 구축을 위한 질의 기반의 문서 요약 기술 연구 및 데이터


## **KorWikiTriple**

한국어 Wikidata기반의 triple 데이터셋입니다.

|  | # of data |
|-------------|-------------|
| train | 79028 |
| dev | 16965 |
| test | 29465 |


## **QSG-Transformer**

상호참조해결과 의존 구조 분석기를 통한 질의 기반의 문서 생성 요약 모델

This repository contains code corresponding to the paper **QSG Transformer: Transformer with Query-Attentive Semantic Graph for Query-Focused Summarization** by Choongwon Park and Youngjoong Ko, published at SIGIR 2022 [\[pdf\]].

**Setup**
```
sudo docker build -t qsg-transformer:latest -f DockerFile .
sudo docker run -itd --gpus all --name qsg-transformer -v $PWD:/QSG-Transformer -p 6006:6006 qsg-transformer:latest
```

**Preprocessing**
```
cd /QSG-Transformer/data
python preprocess.py
```

**Running**
```
cd /QSG-Transformer
python main.py
```

**Tensorboard**
```
tensorboard --logdir=/QSG-Transformer/output --port=6006 --host=0.0.0.0
```
[\[pdf\]]:https://dl.acm.org/doi/pdf/10.1145/3477495.3531901
