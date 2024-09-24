import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import set_seed
from dataset import *
from model import *


class SingleLabelBERTTrainer:
    def __init__(self, model_name, tokenizer, criterion, device, seed=2023):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = device

    def training_cv(
        self,
        df,
        label_name,
        cv,
        batch_size=32,
        epochs=10,
        learning_rate=2e-5,
        max_length=512,
        early_stopping_rounds=5,
        label2id=None,
    ):
        cv_labels = df[label_name].values
        cv_preds = np.zeros((len(df)))
        finished_epochs = []
        for i, (idx_train, idx_eval) in enumerate(cv):
            train_folds = df.loc[idx_train].reset_index(drop=True)
            eval_folds = df.loc[idx_eval].reset_index(drop=True)
            train_dataset = SingleLabelDataset(
                self.tokenizer,
                texts=train_folds["text"].to_list(),
                label=train_folds[label_name].to_list(),
                max_length=max_length,
                label2id=label2id,
            )
            eval_dataset = SingleLabelDataset(
                self.tokenizer,
                texts=eval_folds["text"].to_list(),
                label=eval_folds[label_name].to_list(),
                max_length=max_length,
                label2id=label2id,
            )
            self.train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True
            )
            self.eval_loader = DataLoader(
                dataset=eval_dataset, batch_size=batch_size, shuffle=False
            )

            self.model = BERTSingleLabelModel(
                model_name=self.model_name, output_dim=np.unique(cv_labels).size
            ).to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

            # print(f"--------------------Fold{i+1}--------------------")
            best_eval_result = {"loss": np.inf, "f1": 0, "accuracy": 0}
            early_stopping_count = 0
            for epoch in range(epochs):
                # print(f"--------------------Epoch{epoch+1}--------------------")
                train_result = self._train_epoch()
                eval_result = self._eval_epoch()

                # print(f'Train | Loss: {train_result["loss"]:.4f} macro F1: {train_result["f1"]:.4f} Accuracy: {train_result["accuracy"]:.4f}')
                # print(f'Eval | Loss: {eval_result["loss"]:.4f} macro F1: {eval_result["f1"]:.4f} Accuracy: {eval_result["accuracy"]:.4f}')

                if eval_result["loss"] < best_eval_result["loss"]:
                    best_eval_result = eval_result
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1

                if early_stopping_count == early_stopping_rounds:
                    epoch -= early_stopping_rounds
                    break

            finished_epochs.append(epoch + 1)
            cv_preds[idx_eval] = best_eval_result["preds"]

        if label2id:
            id2label = {v: k for k, v in label2id.items()}
            cv_preds = [id2label[v] for v in cv_preds]

        cv_f1 = f1_score(cv_labels, cv_preds, average="macro")
        cv_acc = accuracy_score(cv_labels, cv_preds)
        # print(f"--------------------CV--------------------")
        print(f"BEST Epochs: {finished_epochs}")
        print(f"macro F1: {cv_f1:.4f} Accuracy: {cv_acc:.4f}")

        return cv_preds

    def training(
        self,
        df,
        label_name,
        batch_size=32,
        epochs=10,
        learning_rate=2e-5,
        max_length=512,
        label2id=None,
    ):
        train_dataset = SingleLabelDataset(
            self.tokenizer,
            texts=df["text"].to_list(),
            label=df[label_name].to_list(),
            max_length=max_length,
            label2id=label2id,
        )
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        self.model = BERTSingleLabelModel(
            model_name=self.model_name, output_dim=df[label_name].nunique()
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            train_result = self._train_epoch()

        preds = train_result["preds"]
        if label2id:
            id2label = {v: k for k, v in label2id.items()}
            preds = [id2label[v] for v in preds]

        return preds, self.model

    def inferencing(self, df, model, batch_size=32, max_length=512, label2id=None):
        test_dataset = SingleLabelDataset(
            self.tokenizer,
            texts=df["text"].to_list(),
            max_length=max_length,
            train=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )
        preds = np.empty(0)
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                ids = batch["ids"].to(self.device)
                mask = batch["mask"].to(self.device)

                output = model(ids, mask)
                preds = np.concatenate(
                    (preds, output.cpu().detach().numpy().argmax(axis=1))
                )

        if label2id:
            id2label = {v: k for k, v in label2id.items()}
            preds = [id2label[v] for v in preds]

        return preds

    def _train_epoch(self):
        train_labels = np.empty(0)
        train_preds = np.empty(0)
        train_loss = 0
        self.model.train()
        for step, batch in enumerate(self.train_loader):
            ids = batch["ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            label = batch["label"].to(self.device)

            output = self.model(ids, mask)

            loss = self.criterion(output, label)
            train_loss += loss.item() / len(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_labels = np.concatenate((train_labels, label.cpu().detach().numpy()))
            train_preds = np.concatenate(
                (train_preds, output.cpu().detach().numpy().argmax(axis=1))
            )

        train_f1 = f1_score(train_labels, train_preds, average="macro")
        train_acc = accuracy_score(train_labels, train_preds)

        return {
            "loss": train_loss,
            "f1": train_f1,
            "accuracy": train_acc,
            "labels": train_labels,
            "preds": train_preds,
        }

    def _eval_epoch(self):
        eval_labels = np.empty(0)
        eval_preds = np.empty(0)
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.eval_loader):
                ids = batch["ids"].to(self.device)
                mask = batch["mask"].to(self.device)
                label = batch["label"].to(self.device)

                output = self.model(ids, mask)

                loss = self.criterion(output, label)
                eval_loss += loss.item() / len(batch)

                eval_labels = np.concatenate(
                    (eval_labels, label.cpu().detach().numpy())
                )
                eval_preds = np.concatenate(
                    (eval_preds, output.cpu().detach().numpy().argmax(axis=1))
                )

            eval_f1 = f1_score(eval_labels, eval_preds, average="macro")
            eval_acc = accuracy_score(eval_labels, eval_preds)

            return {
                "loss": eval_loss,
                "f1": eval_f1,
                "accuracy": eval_acc,
                "labels": eval_labels,
                "preds": eval_preds,
            }


class MultiLabelBERTTrainer:
    def __init__(self, model_name, tokenizer, criterion, device, seed=2023):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = device

    def training_cv(
        self,
        df,
        cv,
        batch_size=32,
        epochs=10,
        learning_rate=2e-5,
        max_length=512,
        early_stopping_rounds=5,
        label2id=None,
    ):
        cv_labelsT, cv_labelsN, cv_labelsM = (
            df["T"].values,
            df["N"].values,
            df["M"].values,
        )
        cv_predsT, cv_predsN, cv_predsM = (
            np.zeros((len(df))),
            np.zeros((len(df))),
            np.zeros((len(df))),
        )
        finished_epochs = []
        for i, (idx_train, idx_eval) in enumerate(cv):
            train_folds = df.loc[idx_train].reset_index(drop=True)
            eval_folds = df.loc[idx_eval].reset_index(drop=True)
            train_dataset = MultiLabelDataset(
                self.tokenizer,
                texts=train_folds["text"].to_list(),
                labelT=train_folds["T"].to_list(),
                labelN=train_folds["N"].to_list(),
                labelM=train_folds["M"].to_list(),
                max_length=max_length,
            )
            eval_dataset = MultiLabelDataset(
                self.tokenizer,
                texts=eval_folds["text"].to_list(),
                labelT=eval_folds["T"].to_list(),
                labelN=eval_folds["N"].to_list(),
                labelM=eval_folds["M"].to_list(),
                max_length=max_length,
            )
            self.train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True
            )
            self.eval_loader = DataLoader(
                dataset=eval_dataset, batch_size=batch_size, shuffle=False
            )

            self.model = BERTMultiLabelModel(model_name=self.model_name).to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

            # print(f"--------------------Fold{i+1}--------------------")
            best_eval_result = {"loss": np.inf, "f1": 0, "accuracy": 0}
            early_stopping_count = 0
            for epoch in range(epochs):
                # print(f"--------------------Epoch{epoch+1}--------------------")
                train_result = self._train_epoch()
                eval_result = self._eval_epoch()

                # print(f'Train | Loss: {train_result["loss"]:.4f} macro F1: {train_result["f1"]:.4f} Accuracy: {train_result["accuracy"]:.4f}')
                # print(f'Eval | Loss: {eval_result["loss"]:.4f} macro F1: {eval_result["f1"]:.4f} Accuracy: {eval_result["accuracy"]:.4f}')

                if eval_result["loss"] < best_eval_result["loss"]:
                    best_eval_result = eval_result
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1

                if early_stopping_count == early_stopping_rounds:
                    epoch -= early_stopping_rounds
                    break

            finished_epochs.append(epoch + 1)
            cv_predsT[idx_eval] = best_eval_result["predsT"]
            cv_predsN[idx_eval] = best_eval_result["predsN"]
            cv_predsM[idx_eval] = best_eval_result["predsM"]

        id2labelT = {0: 1, 1: 2, 2: 3, 3: 4}
        cv_predsT = [id2labelT[v] for v in cv_predsT]

        cv_f1T = f1_score(cv_labelsT, cv_predsT, average="macro")
        cv_f1N = f1_score(cv_labelsN, cv_predsN, average="macro")
        cv_f1M = f1_score(cv_labelsM, cv_predsM, average="macro")
        cv_f1 = (cv_f1T + cv_f1N + cv_f1M) / 3
        cv_accT = accuracy_score(cv_labelsT, cv_predsT)
        cv_accN = accuracy_score(cv_labelsN, cv_predsN)
        cv_accM = accuracy_score(cv_labelsM, cv_predsM)
        cv_acc = (cv_accT + cv_accN + cv_accM) / 3
        # print(f"--------------------CV--------------------")
        print(f"BEST Epochs: {finished_epochs}")
        print(f"macro F1: {cv_f1:.4f} Accuracy: {cv_acc:.4f}")
        print(f"T | macro F1: {cv_f1T:.4f} Accuracy: {cv_accT:.4f}")
        print(f"N | macro F1: {cv_f1N:.4f} Accuracy: {cv_accN:.4f}")
        print(f"M | macro F1: {cv_f1M:.4f} Accuracy: {cv_accM:.4f}")

        cv_predsTNM = [
            str(int(t)) + str(int(n)) + str(int(m))
            for t, n, m in zip(cv_predsT, cv_predsN, cv_predsM)
        ]

        return cv_predsTNM

    def training(
        self, df, batch_size=32, epochs=10, learning_rate=2e-5, max_length=512
    ):
        train_dataset = MultiLabelDataset(
            self.tokenizer,
            texts=df["text"].to_list(),
            labelT=df["T"].to_list(),
            labelN=df["N"].to_list(),
            labelM=df["M"].to_list(),
            max_length=max_length,
        )
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        self.model = BERTMultiLabelModel(model_name=self.model_name).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            train_result = self._train_epoch()

        predsT = train_result["predsT"]
        predsN = train_result["predsN"]
        predsM = train_result["predsM"]

        id2labelT = {0: 1, 1: 2, 2: 3, 3: 4}
        predsT = [id2labelT[v] for v in predsT]

        preds = [
            str(int(t)) + str(int(n)) + str(int(m))
            for t, n, m in zip(predsT, predsN, predsM)
        ]

        return preds, self.model

    def inferencing(self, df, model, batch_size=32, max_length=512, label2id=None):
        test_dataset = MultiLabelDataset(
            self.tokenizer,
            texts=df["text"].to_list(),
            max_length=max_length,
            train=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )
        predsT, predsN, predsM = np.empty(0), np.empty(0), np.empty(0)
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                ids = batch["ids"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputT, outputN, outputM = model(ids, mask)
                predsT = np.concatenate(
                    (predsT, outputT.cpu().detach().numpy().argmax(axis=1))
                )
                predsN = np.concatenate(
                    (predsN, outputN.cpu().detach().numpy().argmax(axis=1))
                )
                predsM = np.concatenate(
                    (predsM, outputM.cpu().detach().numpy().argmax(axis=1))
                )

        id2labelT = {0: 1, 1: 2, 2: 3, 3: 4}
        predsT = [id2labelT[v] for v in predsT]

        preds = [
            str(int(t)) + str(int(n)) + str(int(m))
            for t, n, m in zip(predsT, predsN, predsM)
        ]

        return preds

    def _train_epoch(self):
        train_labelsT, train_labelsN, train_labelsM = (
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
        train_predsT, train_predsN, train_predsM = np.empty(0), np.empty(0), np.empty(0)
        train_loss = 0
        self.model.train()
        for step, batch in enumerate(self.train_loader):
            ids = batch["ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            labelT = batch["labelT"].to(self.device)
            labelN = batch["labelN"].to(self.device)
            labelM = batch["labelM"].to(self.device)

            outputT, outputN, outputM = self.model(ids, mask)

            lossT = self.criterion(outputT, labelT)
            lossN = self.criterion(outputN, labelN)
            lossM = self.criterion(outputM, labelM)
            loss = (lossT + lossN + lossM) / 3
            train_loss += loss.item() / len(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_labelsT = np.concatenate(
                (train_labelsT, labelT.cpu().detach().numpy())
            )
            train_labelsN = np.concatenate(
                (train_labelsN, labelN.cpu().detach().numpy())
            )
            train_labelsM = np.concatenate(
                (train_labelsM, labelM.cpu().detach().numpy())
            )
            train_predsT = np.concatenate(
                (train_predsT, outputT.cpu().detach().numpy().argmax(axis=1))
            )
            train_predsN = np.concatenate(
                (train_predsN, outputN.cpu().detach().numpy().argmax(axis=1))
            )
            train_predsM = np.concatenate(
                (train_predsM, outputM.cpu().detach().numpy().argmax(axis=1))
            )

        train_f1T = f1_score(train_labelsT, train_predsT, average="macro")
        train_f1N = f1_score(train_labelsN, train_predsN, average="macro")
        train_f1M = f1_score(train_labelsM, train_predsM, average="macro")
        train_f1 = (train_f1T + train_f1N + train_f1M) / 3
        train_accT = accuracy_score(train_labelsT, train_predsT)
        train_accN = accuracy_score(train_labelsN, train_predsN)
        train_accM = accuracy_score(train_labelsM, train_predsM)
        train_acc = (train_accT + train_accN + train_accM) / 3

        return {
            "loss": train_loss,
            "f1": train_f1,
            "accuracy": train_acc,
            "labelsT": train_labelsT,
            "labelsN": train_labelsN,
            "labelsM": train_labelsM,
            "predsT": train_predsT,
            "predsN": train_predsN,
            "predsM": train_predsM,
        }

    def _eval_epoch(self):
        eval_labelsT, eval_labelsN, eval_labelsM = np.empty(0), np.empty(0), np.empty(0)
        eval_predsT, eval_predsN, eval_predsM = np.empty(0), np.empty(0), np.empty(0)
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.eval_loader):
                ids = batch["ids"].to(self.device)
                mask = batch["mask"].to(self.device)
                labelT = batch["labelT"].to(self.device)
                labelN = batch["labelN"].to(self.device)
                labelM = batch["labelM"].to(self.device)

                outputT, outputN, outputM = self.model(ids, mask)

                lossT = self.criterion(outputT, labelT)
                lossN = self.criterion(outputN, labelN)
                lossM = self.criterion(outputM, labelM)
                loss = (lossT + lossN + lossM) / 3
                eval_loss += loss.item() / len(batch)

                eval_labelsT = np.concatenate(
                    (eval_labelsT, labelT.cpu().detach().numpy())
                )
                eval_labelsN = np.concatenate(
                    (eval_labelsN, labelN.cpu().detach().numpy())
                )
                eval_labelsM = np.concatenate(
                    (eval_labelsM, labelM.cpu().detach().numpy())
                )
                eval_predsT = np.concatenate(
                    (eval_predsT, outputT.cpu().detach().numpy().argmax(axis=1))
                )
                eval_predsN = np.concatenate(
                    (eval_predsN, outputN.cpu().detach().numpy().argmax(axis=1))
                )
                eval_predsM = np.concatenate(
                    (eval_predsM, outputM.cpu().detach().numpy().argmax(axis=1))
                )

            eval_f1T = f1_score(eval_labelsT, eval_predsT, average="macro")
            eval_f1N = f1_score(eval_labelsN, eval_predsN, average="macro")
            eval_f1M = f1_score(eval_labelsM, eval_predsM, average="macro")
            eval_f1 = (eval_f1T + eval_f1N + eval_f1M) / 3
            eval_accT = accuracy_score(eval_labelsT, eval_predsT)
            eval_accN = accuracy_score(eval_labelsN, eval_predsN)
            eval_accM = accuracy_score(eval_labelsM, eval_predsM)
            eval_acc = (eval_accT + eval_accN + eval_accM) / 3

            return {
                "loss": eval_loss,
                "f1": eval_f1,
                "accuracy": eval_acc,
                "labelsT": eval_labelsT,
                "labelsN": eval_labelsN,
                "labelsM": eval_labelsM,
                "predsT": eval_predsT,
                "predsN": eval_predsN,
                "predsM": eval_predsM,
            }
