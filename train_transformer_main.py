import torch
import pandas as pd
from SMILESTokenizer import SMILESTokenizer
from HIVMoleculeTransformer import HIVMoleculeTransformer
from transformer_trainer import TransformerTrainer

# Data parameters
TRAIN_PATH = 'dataset\HIV_train.csv'
VALIDATION_PATH = 'dataset\HIV_validation.csv'

# Transformer parameters
NUM_HEADS = 4
NUM_LAYERS = 3
EMBEDDING_DIM = 128

# Tokenizer parameters
MAX_LEN = 100

# Training parameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
CRITERION_NAME = 'CrossEntropyLoss'
OPTIMIZER_NAME = 'Adam'
IMBALANCE_METHOD = 'weighted_loss'

SAVING_PATH = 'models\hiv_transformer_sampled_20epoch.pth'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    df = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VALIDATION_PATH)

    tokenizer = SMILESTokenizer()

    print("Building vocabulary...")
    tokenizer.build_vocabulary(df["smiles"].tolist())
    vocab_size = len(tokenizer.stoi)
    pad_idx = tokenizer.stoi[tokenizer.pad_token]



    model = HIVMoleculeTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        num_classes=2,
        pad_idx=pad_idx
    ).to(device)

    trainer = TransformerTrainer(transformer=model, tokenizer=tokenizer, df=df, validation_df=df_val, device=device)

    model = trainer.train(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, max_len=MAX_LEN,
                          criterion_name=CRITERION_NAME, optimizer_name=OPTIMIZER_NAME, imbalance_method=IMBALANCE_METHOD)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_stoi': tokenizer.stoi,
        'vocab_itos': tokenizer.itos,
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'max_len': MAX_LEN,
            'num_classes': 2,
            'pad_idx': pad_idx
        }
    }

    torch.save(checkpoint, SAVING_PATH)
    print(f"Model saved to '{SAVING_PATH}'")

if __name__ == '__main__':
    main()