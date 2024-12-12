import torch.nn as nn
import torch

class RegularizedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, dropout=0.2):
        super(RegularizedLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        # This is new!
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(positions)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        logits = self.linear(embeddings)
        return logits


class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super(SimpleLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(positions)
        
        embeddings = token_embeds + position_embeds
        logits = self.linear(embeddings)
        return logits


def generate_text(model, tokenizer, start_text, device, context_length=15, temperature=1.0):
    model.eval()
    generated = tokenizer.encode(start_text)
    context = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(context_length):
            if context.size(1) >= context_length:
                break
            logits = model(context)
            next_token_logits = logits[0, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            context = torch.cat([context, next_token_id.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(context[0].tolist())
    return generated_text