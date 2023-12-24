import torch
from torch import nn
from flair.embeddings import TransformerDocumentEmbeddings


# For embedding generation
model = TransformerDocumentEmbeddings('ogimgio/K-12BERT-reward-neurallinguisticpioneers-3', fine_tune=True, layers="-1")

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model, hidden_size=256):
        super().__init__()
        self.embedding_model = embedding_model
        self.fc = nn.Linear(embedding_model.embedding_length, hidden_size) # embedding_model.embedding_length * 2 # With torch.cat() instead of stack()
        self.gelu = nn.GELU()
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        # Obtain embeddings for both inputs
        self.embedding_model.embed(input1)
        self.embedding_model.embed(input2)
        embeddings1 = input1.embedding # torch.stack([i.embedding for i in input1]) # When batching
        embeddings2 = input2.embedding # torch.stack([i.embedding for i in input2])

        # Concatenate embeddings
        concatenated = torch.stack((embeddings1, embeddings2))
        # FIXME check shape & add batch norm | NOTE: OOM, so ignoring batching

        # Pass through the Siamese network
        x = self.fc(concatenated) # 2 x 512
        x = self.gelu(x)
        x = self.out(x) # 2 x 1
        x = self.sigmoid(x.squeeze()) # No squeeze() with torch.cat()

        return torch.mean(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output, label):
        loss = label * torch.pow(output, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - output, min=0.0), 2)

        return torch.mean(loss)
