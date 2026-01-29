from Tokenization.SMILESTokenizer import SMILESTokenizer
import pandas as pd

csv_file_path = '../Data/HIV_train.csv'
saving_path = '../Vocabulary/first_vocab'

df = pd.read_csv(csv_file_path)

tokenizer = SMILESTokenizer()

clean_smiles = [tokenizer.canonicalize(s) for s in df['smiles'] if tokenizer.canonicalize(s)]
tokenizer.build_vocabulary(clean_smiles)

tokenizer.save_vocab(saving_path)

