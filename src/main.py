import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertJapaneseTokenizer
import fire

# from utils import load_dataset
from train import SingleLabelBERTTrainer, MultiLabelBERTTrainer


def main(
    model_type="single",
    model_name="cl-tohoku/bert-base-japanese-v2",
    train_data_path="data/data.csv",
    seed=2023,
    folds=5,
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    batch_size=32,
    learning_rate=2e-5,
    epochs=100,
    max_length=512,
    early_stopping_rounds=5,
    criterion=nn.CrossEntropyLoss(),
):
    print("Model Type: ", model_type)
    # Read Data
    # df = load_dataset(train_data_dir)
    df = pd.read_csv(train_data_path)
    df["TNM"] = df["T"].astype(str) + df["N"].astype(str) + df["M"].astype(str)
    fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labelTNM2id = {v: i for i, v in enumerate(df["TNM"].unique())}

    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    label_name = "TNM"
    cv = fold.split(df["text"], df["TNM"])

    if model_type == "single":
        trainer = SingleLabelBERTTrainer(
            model_name=model_name,
            tokenizer=TOKENIZER,
            criterion=criterion,
            device=device,
            seed=seed,
        )
        cs_predsTNM = trainer.training_cv(
            df,
            label_name,
            cv,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            early_stopping_rounds=early_stopping_rounds,
            label2id=labelTNM2id,
        )
    else:
        trainer = MultiLabelBERTTrainer(
            model_name=model_name,
            tokenizer=TOKENIZER,
            criterion=criterion,
            device=device,
            seed=seed,
        )
        cs_predsTNM = trainer.training_cv(
            df,
            cv,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            early_stopping_rounds=early_stopping_rounds,
        )
    df["TNM_pred"] = cs_predsTNM

    df.to_csv(f"outputs/{model_type}_predicted.csv", index=False)
    return cs_predsTNM


if __name__ == "__main__":
    fire.Fire(main)
