# **QSG Transformer: Transformer with Query-Attentive Semantic Graph for Query-Focused Summarization**

This repository contains code corresponding to the paper **QSG Transformer: Transformer with Query-Attentive Semantic Graph for Query-Focused Summarization** by Choongwon Park and Youngjoong Ko, published at SIGIR 2022 [\[pdf\]].

## **Setup**
```
sudo docker build -t qsg-transformer:latest -f DockerFile .
sudo docker run -itd --gpus all --name qsg-transformer -v $PWD:/QSG-Transformer -p 6006:6006 qsg-transformer:latest
```

## **Preprocessing**
```
cd /QSG-Transformer/data
python preprocess.py
```

## **Running**
```
cd /QSG-Transformer
python main.py
```

## **Tensorboard**
```
tensorboard --logdir=/QSG-Transformer/output --port=6006 --host=0.0.0.0
```
[\[pdf\]]:https://dl.acm.org/doi/pdf/10.1145/3477495.3531901