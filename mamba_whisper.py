"""
State Space Model for Speech Recognition (Mamba-Whisper Style)

A simplified implementation of a state space model for automatic speech recognition,
inspired by Mamba SSM and Whisper architectures.

Based on:
- Mamba: Linear-time Sequence Modeling with Selective State Spaces
- Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
- Samba-ASR: State-of-the-Art Speech Recognition with SSMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SSMBlock(nn.Module):
    """
    State Space Model Block (Simplified Mamba-style)
    
    A simplified SSM using gating mechanism instead of full state space.
    For production, use the mamba package for proper SSM implementation.
    """
    def __init__(self, d_model, d_state=16, conv_kernel=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Gated linear unit
        self.gate = nn.Linear(d_model, d_model * 2)
        
        # State projection
        self.state_proj = nn.Linear(d_model, d_state)
        self.state_out = nn.Linear(d_state, d_model)
        
        # Convolution for local context
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel//2)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, training=True):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.norm(x)
        
        # Convolution branch for local context
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Gated linear unit
        gate_input = self.gate(x_conv)
        gate, value = gate_input.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        
        # State computation
        state = F.silu(self.state_proj(x_conv))
        state_out = self.state_out(state)
        
        # Gated output
        y = gate * value + (1 - gate) * state_out
        
        # Output projection
        y = self.out_proj(y)
        if training:
            y = self.dropout(y)
        
        return y + residual


class MultiHeadAttention(nn.Module):
    """Multi-head attention for cross-attention (like Whisper)"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class EncoderBlock(nn.Module):
    """Encoder block with SSM and feed-forward"""
    def __init__(self, d_model, d_state=16, d_ff=None, num_heads=4, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.ssm = SSMBlock(d_model, d_state, dropout=dropout)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
    
    def forward(self, x, mask=None, training=True):
        x = x + self.ssm(x, training=training)
        x = self.norm1(x + self.attention(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block with SSM and cross-attention"""
    def __init__(self, d_model, d_state=16, d_ff=None, num_heads=4, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.ssm = SSMBlock(d_model, d_state, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, training=True):
        x = x + self.ssm(x, training=training)
        x = self.norm1(x + self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.norm2(x + self.ffn(x))
        return x


class AudioEncoder(nn.Module):
    """Audio encoder (mel spectrogram -> embeddings)"""
    def __init__(self, n_mels=80, d_model=384):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv1d(n_mels, d_model // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 3000, d_model))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, mel_spec):
        # mel_spec: (batch, n_mels, time)
        x = F.gelu(self.conv1(mel_spec))
        x = F.gelu(self.conv2(x))
        
        x = x.transpose(1, 2)  # (batch, time, d_model)
        
        # Add positional embedding
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        return self.dropout(x)


class TextDecoder(nn.Module):
    """Text decoder for speech recognition"""
    def __init__(self, vocab_size=51865, d_model=384, d_state=16, d_ff=None,
                 num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2000, d_model))
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_state, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, encoder_output, target_ids, tgt_mask=None, training=True):
        # target_ids: (batch, seq_len)
        seq_len = target_ids.shape[1]
        x = self.token_embedding(target_ids) + self.pos_embedding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, training=training)
        
        x = self.norm(x)
        return self.lm_head(x)
    
    def generate(self, encoder_output, max_length=100, eos_token_id=50257, temperature=1.0):
        """Generate text using greedy decoding"""
        device = next(self.parameters()).device
        batch_size = encoder_output.shape[0]
        
        # Start with begin token
        target_ids = torch.full((batch_size, 1), 50256, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            tgt_mask = self._causal_mask(target_ids.shape[1]).to(device)
            logits = self.forward(encoder_output, target_ids, tgt_mask, training=False)
            
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            target_ids = torch.cat([target_ids, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        return target_ids
    
    def _causal_mask(self, size):
        return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


class MambaWhisperASR(nn.Module):
    """
    State Space Model for Speech Recognition
    
    Architecture:
    - Audio Encoder: Mel spectrogram -> features
    - Encoder: Stack of SSM blocks
    - Decoder: Stack of SSM + cross-attention blocks
    
    Like Whisper but using Mamba SSM instead of Transformers.
    """
    def __init__(
        self,
        vocab_size=51865,
        d_model=384,
        d_state=16,
        d_ff=None,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=4,
        n_mels=80,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Audio encoder
        self.encoder = AudioEncoder(n_mels=n_mels, d_model=d_model)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, d_state, d_ff, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = RMSNorm(d_model)
        
        # Text decoder
        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=d_state,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
    
    def forward(self, mel_spec, target_ids, tgt_mask=None, training=True):
        # Encode audio
        encoder_output = self.encoder(mel_spec)
        
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, training=training)
        
        encoder_output = self.encoder_norm(encoder_output)
        
        # Decode text
        logits = self.decoder(encoder_output, target_ids, tgt_mask, training=training)
        
        return logits
    
    def transcribe(self, mel_spec, max_length=100, temperature=1.0):
        """Transcribe audio to text"""
        # Encode audio
        encoder_output = self.encoder(mel_spec)
        
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, training=False)
        
        encoder_output = self.encoder_norm(encoder_output)
        
        # Generate text
        return self.decoder.generate(encoder_output, max_length, temperature=temperature)


def create_causal_mask(size, device='cpu'):
    """Create causal mask for decoder"""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with dummy data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = MambaWhisperASR(
        vocab_size=51865,
        d_model=256,
        d_state=16,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=4,
        n_mels=80
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Dummy mel spectrogram (batch, n_mels, time)
    mel_spec = torch.randn(2, 80, 300).to(device)
    
    # Dummy target IDs
    target_ids = torch.randint(0, 51865, (2, 20)).to(device)
    
    # Create causal mask
    tgt_mask = create_causal_mask(20, device)
    
    # Forward pass
    logits = model(mel_spec, target_ids, tgt_mask)
    print(f"Input mel: {mel_spec.shape}")
    print(f"Target: {target_ids.shape}")
    print(f"Output logits: {logits.shape}")
    
    # Test generation
    generated = model.transcribe(mel_spec, max_length=30)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0, :10].tolist()}")
    
    return model


if __name__ == "__main__":
    test_model()
