import torch
import torch.nn as nn

class GELUTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 num_heads=4,
                 num_layers=3,
                 max_len=100,
                 num_classes=2,
                 pad_idx=0):
        super().__init__()

        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        padding_mask = (x == 0)

        x = self.embedding(x)

        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        output = self.transformer(x, src_key_padding_mask=padding_mask)

        cls_token_output = output[:, 0, :]

        logits = self.classifier(cls_token_output)

        return logits
