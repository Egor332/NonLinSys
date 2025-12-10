import re
from rdkit import Chem
import torch
import os
import ast


class SMILESTokenizer:
    def __init__(self):
        self.atom_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.atom_pattern)

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"

        self.stoi = {}
        self.itos = {}

    def canonicalize(self, smiles):
        """Standardizes SMILES using RDKit"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return None

    def tokenize(self, smiles):
        """Splits SMILES into list of atom tokens"""
        return [token for token in self.regex.findall(smiles)]

    def build_vocabulary(self, smiles_list):
        """Creates the dictionary from a list of SMILES"""
        unique_tokens = set()
        for s in smiles_list:
            tokens = self.tokenize(s)
            unique_tokens.update(tokens)

        sorted_tokens = sorted(list(unique_tokens))

        vocab = [self.pad_token, self.unk_token, self.cls_token, self.sep_token] + sorted_tokens

        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for idx, token in enumerate(vocab)}

        print(f"Vocabulary built. Size: {len(self.stoi)} tokens.")
        return self.stoi

    def save_vocab(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, 'vocab_itos.txt'), 'w') as f:
            f.write(str(self.itos))
        with open(os.path.join(dirpath, 'vocab_stoi.txt'), 'w') as f:
            f.write(str(self.stoi))

    def load_vocab(self, dirpath):
        itos_path = os.path.join(dirpath, 'vocab_itos.txt')
        stoi_path = os.path.join(dirpath, 'vocab_stoi.txt')

        loaded_vocab = {}

        try:
            with open(itos_path, 'r') as f:
                itos_str = f.read()
                loaded_vocab['itos'] = ast.literal_eval(itos_str)
        except FileNotFoundError:
            print(f"Error: ITOS file not found at {itos_path}")
            return None

        try:
            with open(stoi_path, 'r') as f:
                stoi_str = f.read()
                loaded_vocab['stoi'] = ast.literal_eval(stoi_str)
        except FileNotFoundError:
            print(f"Error: STOI file not found at {stoi_path}")
            return None

        print(f"Successfully loaded vocabularies from {dirpath}.")
        self.itos = loaded_vocab['itos']
        self.stoi = loaded_vocab['stoi']


    def encode(self, smiles, max_len=100):
        """Converts SMILES string to Padded Tensor of Integers"""
        canon_smiles = self.canonicalize(smiles)
        if canon_smiles is None:
            return None

        tokens = self.tokenize(canon_smiles)

        tokens = [self.cls_token] + tokens + [self.sep_token]

        ids = [self.stoi.get(t, self.stoi[self.unk_token]) for t in tokens]

        if len(ids) < max_len:
            ids = ids + [self.stoi[self.pad_token]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return torch.tensor(ids, dtype=torch.long)