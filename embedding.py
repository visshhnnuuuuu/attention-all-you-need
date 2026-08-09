#converting the token id into embedding vector
import torch
from tokenizer import Tokenizer
import torch.nn as nn
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size,embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)



tokenizer = Tokenizer()
token_ids=tokenizer.encode_text("Hello, world!")
token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
vocab_size = tokenizer.encoding.n_vocab # Size of the vocabulary
embedding_dim = 128  # Dimension of the embedding vectors
embedding_model = EmbeddingModel(vocab_size, embedding_dim)
embedded_tokens = embedding_model(token_ids_tensor)    
print(embedded_tokens)