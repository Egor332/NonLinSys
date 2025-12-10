import torch

class HIVDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=100):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['smiles']
        label = row['HIV_active']

        token_ids = self.tokenizer.encode(smiles, self.max_len)

        if token_ids is None:
            token_ids = torch.zeros(self.max_len, dtype=torch.long)

        return token_ids, torch.tensor(label, dtype=torch.long)