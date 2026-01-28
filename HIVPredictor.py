import torch
import torch.nn as nn
from SMILESTokenizer import SMILESTokenizer
from HIVMoleculeTransformer import HIVMoleculeTransformer
from GELU_transformer import GELUTransformer

class HIVPredictor:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on: {self.device}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.tokenizer = SMILESTokenizer()
        self.tokenizer.stoi = checkpoint["vocab_stoi"]
        self.tokenizer.itos = checkpoint["vocab_itos"]
        self.max_len = checkpoint["config"]["max_len"]

        config = checkpoint["config"]
        self.model = GELUTransformer(
            vocab_size=config['vocab_size'],
            embed_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            max_len=config['max_len'],
            num_classes=config['num_classes'],
            pad_idx=config['pad_idx']
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def predict(self, smiles):
        token_ids = self.tokenizer.encode(smiles, self.max_len)

        if token_ids is None:
            return {"error": "Incorrect SMILES"}

        input_tensor = token_ids.unsqueeze(0).to(self.device)

        with torch.no_grad():

            if self.model.num_classes == 1:
                logits = self.model(input_tensor)
                active_prob = torch.sigmoid(logits).item()
                inactive_prob = 1.0 - active_prob
                predicted_class = 1 if active_prob >= 0.5 else 0
            else:
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                inactive_prob = probs[0][0].item()
                active_prob = probs[0][1].item()
                # Determine class
                predicted_class = torch.argmax(probs, dim=1).item()

        return {
            "smiles": smiles,
            "is_active": bool(predicted_class == 1),
            "confidence_active": f"{active_prob:.4f}",
            "confidence_inactive": f"{inactive_prob:.4f}"
        }

