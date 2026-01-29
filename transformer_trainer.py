import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from Data.HIVDataset import HIVDataset
from Transformers.binary_focal_loss import BinaryFocalLoss

class TransformerTrainer:
    def __init__(self, transformer, tokenizer, df, validation_df, device):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.df = df
        self.validation_df = validation_df
        self.device = device

    def train(self, batch_size, learning_rate, epochs, max_len, criterion_name="CrossEntropyLoss",
              optimizer_name="Adam", imbalance_method="weighted_loss"):  # Options: 'weighted_loss', 'sampler', 'none'

        dataset = HIVDataset(self.df, self.tokenizer, max_len=max_len)
        validation_dataset = HIVDataset(self.validation_df, self.tokenizer, max_len=max_len)


        sampler = None
        shuffle = True

        weight_ce = None
        pos_weight_bce = None

        targets = self.df["HIV_active"].values
        neg_count = len(targets) - sum(targets)
        pos_count = sum(targets)

        print(f"Imbalance Strategy: {imbalance_method}")


        if imbalance_method == "sampler":
            class_counts = [neg_count, pos_count]
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = [class_weights[t] for t in targets]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        elif imbalance_method == "weighted_loss":
            if pos_count > 0:
                pos_weight_bce = torch.tensor(neg_count / pos_count, dtype=torch.float).to(self.device)

                weight_ce = torch.tensor([1.0, neg_count / pos_count], dtype=torch.float).to(self.device)


        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


        print(f"Weight_ce: {weight_ce}, weight_bce: {pos_weight_bce}")

        criterion = self._get_criterion(criterion_name, weight_ce=weight_ce, pos_weight_bce=pos_weight_bce)
        optimizer = self._get_optimizer(optimizer_name, learning_rate)

        is_binary_loss = criterion_name in ["BCEWithLogitsLoss", "FocalLoss"]

        print("Training started...")

        for epoch in range(epochs):
            self.transformer.train()
            total_loss = 0

            loop = tqdm(train_loader, leave=True)

            for batch_idx, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.transformer(inputs)

                if is_binary_loss:
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                loop.set_postfix(loss=loss.item())


            self.transformer.eval()
            all_targets = []
            all_probs = []

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.transformer(inputs)

                    if is_binary_loss:
                        probs = torch.sigmoid(outputs)
                        probs = probs.squeeze(1)
                    else:
                        probs = torch.softmax(outputs, dim=1)[:, 1]  # Shape [B]

                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())


            try:
                val_auc = roc_auc_score(all_targets, all_probs)
                print(
                    f"End of Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader):.4f} | Val ROC-AUC: {val_auc:.4f}")
            except ValueError:
                print(
                    f"End of Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader):.4f} | Val ROC-AUC: N/A (Only one class in val set)")

        return self.transformer

    def _get_criterion(self, criterion_name, weight_ce=None, pos_weight_bce=None):
        if criterion_name == "CrossEntropyLoss":
            # Standard CE uses a weight vector [w_neg, w_pos]
            return nn.CrossEntropyLoss(weight=weight_ce)

        elif criterion_name == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_bce)

        elif criterion_name == "FocalLoss":
            return BinaryFocalLoss(gamma=2.0, alpha=0.25, pos_weight=pos_weight_bce)

        else:
            print(f"Warning: Criterion {criterion_name} not found. Defaulting to CrossEntropyLoss.")
            return nn.CrossEntropyLoss(weight=weight_ce)

    def _get_optimizer(self, optimizer_name, lr):
        if optimizer_name == "Adam":
            return optim.Adam(self.transformer.parameters(), lr=lr)
        elif optimizer_name == "AdamW":
            return optim.AdamW(self.transformer.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == "SGD":
            return optim.SGD(self.transformer.parameters(), lr=lr, momentum=0.9)
        else:
            print(f"Warning: Optimizer {optimizer_name} not found. Using Adam.")
            return optim.Adam(self.transformer.parameters(), lr=lr)