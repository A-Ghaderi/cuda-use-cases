import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Define the Multi-Head Self-Attention Mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)


    def forward(self, x):
        batch_size, seq_length, d_model = x.size()


        # Linear projections
        q = self.linear_q(x).view(batch_size, seq_length, self.nhead, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_length, self.nhead, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_length, self.nhead, self.d_k).transpose(1, 2)


        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.nn.functional.softmax(scores, dim=-1)


        # Attention values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        return self.fc_out(out)


# Define the Feedforward Neural Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


    def forward(self, x):
        return self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(x))))


# Define a Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)
        self.feed_forward = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        # Self-attention layer
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)


        # Feedforward layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)


        return x


# Define the Transformer Model
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model


    def forward(self, src):
        # Embed and scale
        x = self.embedding(src) * math.sqrt(self.d_model)


        # Pass through each transformer layer
        for layer in self.layers:
            x = layer(x)


        # Final linear layer to output vocabulary size
        return self.fc_out(x)


# Initialize model parameters
vocab_size = 5000  # Example vocabulary size
d_model = 512      # Embedding size
nhead = 8          # Number of attention heads
num_layers = 6     # Number of transformer layers


# Create model
model = SimpleTransformerModel(vocab_size, d_model, nhead, num_layers)


# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("the device is:", device)
model.to(device)


# Sample training loop (using dummy data for demonstration)
def train_model(model, num_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(100):  # Dummy loop for demonstration
            # Dummy data
            inputs = torch.randint(0, vocab_size, (32, 10)).to(device)  # Batch size 32, sequence length 10
            targets = torch.randint(0, vocab_size, (32, 10)).to(device)


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
        
        avg_loss = total_loss / 100  # Dummy loop length
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


# Example usage
train_model(model)
