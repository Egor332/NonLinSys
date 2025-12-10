from SMILESTokenizer import SMILESTokenizer
import pandas as pd

csv_file_path = 'dataset/HIV_train.csv'
df = pd.read_csv(csv_file_path)

tokenizer = SMILESTokenizer()

clean_smiles = [tokenizer.canonicalize(s) for s in df['smiles'] if tokenizer.canonicalize(s)]
tokenizer.build_vocabulary(clean_smiles)

tokenizer.save_vocab("vocabulary/first_vocab")

