# TNM-Classifier
## 説明
BERT で TNM 分類を行うプログラムです．[NTCIR-17 MedNLP-SC Radiology Report Subtask](https://repository.nii.ac.jp/records/2001285)で取り組んだコードを元にしたプログラムです．
[rye](https://rye.astral.sh/guide/installation/)で python のパッケージを管理しています．

## データの概要
|  id  |  text  |  tnm  |
| ---- | ---- | ---- |
|  0  |  左上葉気管支は閉塞して造影  CT  で増強効果の乏しい 70mm  の腫瘤があります。肺癌と考えます。  |  430  |

## 使用方法
Rye を導入した後，rye syncで python ライブラリをインストール<br>

下記のコマンド一覧を参考に適切な引数を追加し，実行する．<br>
実行例
```
rye run python src/main.py -i='data/tnm_report.csv' -t='multi' -n=10 -b=8
```

## コマンド一覧

help が見れるコマンド: `rye run python src/main.py --help` <br>
入力すると以下の説明が表示されます．

```
NAME
    main.py

SYNOPSIS
    main.py <flags>

FLAGS
    -t, --type=TYPE
        Type: str
        Default: 'single'
    -p, --pretrained_model=PRETRAINED_MODEL
        Type: str
        Default: 'cl-tohoku/bert-ba...
    -i, --input_path=INPUT_PATH
        Type: str
        Default: 'data/data.csv'
    -s, --seed=SEED
        Type: int
        Default: 2023
    -f, --folds=FOLDS
        Type: int
        Default: 5
    -d, --device=DEVICE
        Type: str
        Default: device(type='cuda', ind...
    -b, --batch_size=BATCH_SIZE
        Type: int
        Default: 32
    -l, --learning_rate=LEARNING_RATE
        Type: float
        Default: 2e-05
    -n, --num_epochs=NUM_EPOCHS
        Type: int
        Default: 100
    -m, --max_length=MAX_LENGTH
        Type: int
        Default: 512
    -e, --early_stopping_rounds=EARLY_STOPPING_ROUNDS
        Type: int
        Default: 5
    -c, --criterion=CRITERION
        Type: Module
        Default: CrossEntropyLoss()
```
