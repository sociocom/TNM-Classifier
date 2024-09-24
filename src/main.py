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
    type: str = "single",
    pretrained_model: str = "cl-tohoku/bert-base-japanese-v2",
    input_path: str = "data/data.csv",
    seed: int = 2023,
    folds: int = 5,
    device: str = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    num_epochs: int = 100,
    max_length: int = 512,
    early_stopping_rounds: int = 5,
    criterion: nn.Module = nn.CrossEntropyLoss(),
):
    print("Model Type: ", type)
    # Read Data
    # df = load_dataset(train_data_dir)
    df = pd.read_csv(input_path)
    df["TNM"] = df["T"].astype(str) + df["N"].astype(str) + df["M"].astype(str)
    fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labelTNM2id = {v: i for i, v in enumerate(df["TNM"].unique())}

    TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
    label_name = "TNM"
    cv = fold.split(df["text"], df["TNM"])

    if type == "single":
        trainer = SingleLabelBERTTrainer(
            model_name=pretrained_model,
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
            epochs=num_epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            early_stopping_rounds=early_stopping_rounds,
            label2id=labelTNM2id,
        )
    else:
        trainer = MultiLabelBERTTrainer(
            model_name=pretrained_model,
            tokenizer=TOKENIZER,
            criterion=criterion,
            device=device,
            seed=seed,
        )
        cs_predsTNM = trainer.training_cv(
            df,
            cv,
            batch_size=batch_size,
            epochs=num_epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            early_stopping_rounds=early_stopping_rounds,
        )
    df["TNM_pred"] = cs_predsTNM

    df.to_csv(f"outputs/{type}_predicted.csv", index=False)
    return cs_predsTNM


if __name__ == "__main__":
    fire.Fire(main)
