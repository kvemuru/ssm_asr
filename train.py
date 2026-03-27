"""
Training script for Mamba-Whisper ASR model

Supports:
- LibriSpeech dataset (automatic download)
- Training with cross-entropy loss
- Checkpoint saving
- Evaluation on validation set
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=""):
        return x

try:
    import librosa
except ImportError:
    print("Warning: librosa not found, installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'librosa'])
    import librosa


class AudioTokenizer:
    """Simple audio feature extractor (mel spectrogram)"""
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def __call__(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel


class TextTokenizer:
    """Simple text tokenizer (character-level)"""
    def __init__(self):
        self.vocab = list(' abcdefghijklmnopqrstuvwxyz\'.,!?')
        self.token_to_id = {t: i+1 for i, t in enumerate(self.vocab)}
        self.token_to_id['<blank>'] = 0
        self.token_to_id['<eos>'] = len(self.vocab) + 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
    
    def encode(self, text):
        text = text.lower().strip()
        return [self.token_to_id.get(c, 0) for c in text] + [self.token_to_id['<eos>']]
    
    def decode(self, ids):
        text = ''
        for i in ids:
            if i == 0:
                continue
            if i == self.token_to_id['<eos>']:
                break
            text += self.id_to_token.get(i, '')
        return text.strip()


def download_librispeech(data_dir, subset="train"):
    """Download LibriSpeech dataset"""
    import urllib.request
    import tarfile
    
    urls = {
        "train": [
            "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        ],
        "val": [
            "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        ],
        "test": [
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
        ]
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading LibriSpeech {subset} set...")
    
    for url in urls.get(subset, []):
        filename = os.path.join(data_dir, url.split('/')[-1])
        if not os.path.exists(filename.replace('.tar.gz', '')):
            print(f"Downloading {url.split('/')[-1]}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Extracting {filename}...")
                with tarfile.open(filename, 'r:gz') as tar:
                    tar.extractall(data_dir)
                os.remove(filename)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                print("Trying alternative source...")
    
    print("Download complete!")
    return data_dir


def parse_librispeech_transcripts(data_dir, subset="train"):
    """Parse LibriSpeech transcripts"""
    data = []
    
    subset_dirs = {
        "train": ["train-clean-100", "train-clean-360"],
        "val": ["dev-clean"],
        "test": ["test-clean"]
    }
    
    for subdir in subset_dirs.get(subset, []):
        libri_dir = os.path.join(data_dir, subdir)
        if not os.path.exists(libri_dir):
            continue
            
        for speaker in os.listdir(libri_dir):
            speaker_dir = os.path.join(libri_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
                
            for chapter in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                    
                # Find .txt files
                for f in os.listdir(chapter_dir):
                    if f.endswith('.txt'):
                        transcript_file = os.path.join(chapter_dir, f)
                        with open(transcript_file, 'r') as tf:
                            for line in tf:
                                parts = line.strip().split(' ', 1)
                                if len(parts) == 2:
                                    audio_id, transcript = parts
                                    audio_file = os.path.join(chapter_dir, f"{audio_id}.flac")
                                    if os.path.exists(audio_file):
                                        data.append((audio_file, transcript.lower()))
    
    return data


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset"""
    def __init__(self, data_dir, subset="train", audio_tokenizer=None, text_tokenizer=None, 
                 max_audio_len=3000, max_text_len=200, limit=None):
        self.audio_tokenizer = audio_tokenizer or AudioTokenizer()
        self.text_tokenizer = text_tokenizer or TextTokenizer()
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        # Check if data exists
        libri_dir = os.path.join(data_dir, "train-clean-100")
        if not os.path.exists(libri_dir):
            print(f"LibriSpeech not found in {data_dir}, downloading...")
            download_librispeech(data_dir, subset)
        
        # Parse transcripts
        self.data = parse_librispeech_transcripts(data_dir, subset)
        
        if limit:
            self.data = self.data[:limit]
        
        print(f"Loaded {len(self.data)} {subset} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        
        # Load and convert audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract mel spectrogram
        mel = self.audio_tokenizer(audio)
        
        if mel.shape[1] > self.max_audio_len:
            mel = mel[:, :self.max_audio_len]
        
        # Tokenize text
        text_ids = self.text_tokenizer.encode(text)
        if len(text_ids) > self.max_text_len:
            text_ids = text_ids[:self.max_text_len]
        
        return torch.from_numpy(mel).float(), torch.tensor(text_ids, dtype=torch.long)


class SpeechDataset(Dataset):
    """Simple speech recognition dataset"""
    def __init__(self, audio_dir, transcript_file, audio_tokenizer, text_tokenizer, 
                 max_audio_len=3000, max_text_len=200):
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        with open(transcript_file, 'r') as f:
            lines = f.readlines()
        
        self.data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 2:
                audio_path = os.path.join(audio_dir, parts[0].strip())
                text = parts[1].strip()
                if os.path.exists(audio_path):
                    self.data.append((audio_path, text))
        
        print(f"Loaded {len(self.data)} audio-text pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        
        audio, sr = librosa.load(audio_path, sr=16000)
        mel = self.audio_tokenizer(audio)
        
        if mel.shape[1] > self.max_audio_len:
            mel = mel[:, :self.max_audio_len]
        
        text_ids = self.text_tokenizer.encode(text)
        if len(text_ids) > self.max_text_len:
            text_ids = text_ids[:self.max_text_len]
        
        return torch.from_numpy(mel).float(), torch.tensor(text_ids, dtype=torch.long)


def collate_fn(batch):
    """Collate batch with padding"""
    audios, texts = zip(*batch)
    
    # Pad audios
    max_audio_len = max(a.shape[1] for a in audios)
    padded_audios = []
    for a in audios:
        if a.shape[1] < max_audio_len:
            padding = torch.zeros(a.shape[0], max_audio_len - a.shape[1])
            a = torch.cat([a, padding], dim=1)
        padded_audios.append(a)
    audios = torch.stack(padded_audios)
    
    # Pad texts
    max_text_len = max(t.shape[0] for t in texts)
    padded_texts = []
    for t in texts:
        if t.shape[0] < max_text_len:
            padding = torch.zeros(max_text_len - t.shape[0], dtype=torch.long)
            t = torch.cat([t, padding], dim=0)
        padded_texts.append(t)
    texts = torch.stack(padded_texts)
    
    return audios, texts


def create_causal_mask(size, device='cpu'):
    """Create causal mask for decoder"""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (mel_spec, target_ids) in enumerate(pbar):
        mel_spec = mel_spec.to(device)
        target_ids = target_ids.to(device)
        
        tgt_mask = create_causal_mask(target_ids.shape[1], device)
        
        logits = model(mel_spec, target_ids, tgt_mask)
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate(model, dataloader, device, text_tokenizer):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for mel_spec, target_ids in tqdm(dataloader, desc="Evaluating"):
            mel_spec = mel_spec.to(device)
            target_ids = target_ids.to(device)
            
            tgt_mask = create_causal_mask(target_ids.shape[1], device)
            logits = model(mel_spec, target_ids, tgt_mask)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train(args):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tokenizers
    audio_tokenizer = AudioTokenizer()
    text_tokenizer = TextTokenizer()
    print(f"Text vocab size: {text_tokenizer.vocab_size}")
    
    # Create datasets
    if args.dataset == "librispeech":
        print(f"\nLoading LibriSpeech dataset...")
        train_dataset = LibriSpeechDataset(
            data_dir=args.data_dir,
            subset="train",
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=text_tokenizer,
            max_audio_len=args.max_audio_len,
            max_text_len=args.max_text_len,
            limit=args.train_limit
        )
        
        val_dataset = LibriSpeechDataset(
            data_dir=args.data_dir,
            subset="val",
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=text_tokenizer,
            max_audio_len=args.max_audio_len,
            max_text_len=args.max_text_len,
            limit=args.val_limit
        )
    else:
        transcript_file = os.path.join(args.data_dir, "transcripts.txt")
        if not os.path.exists(transcript_file):
            print("Creating synthetic dataset...")
            os.makedirs(args.data_dir, exist_ok=True)
            
            import scipy.io.wavfile as wav
            np.random.seed(42)
            
            sample_texts = [
                "hello world", "this is a test", "speech recognition",
                "machine learning", "deep neural networks",
            ]
            
            for i, text in enumerate(sample_texts):
                duration = 1.0
                sample_rate = 16000
                t = np.linspace(0, duration, int(sample_rate * duration))
                freq = 200 + (i * 50)
                audio = np.sin(2 * np.pi * freq * t) * 0.3
                audio = audio.astype(np.float32)
                wav.write(f"{args.data_dir}/audio_{i:04d}.wav", sample_rate, audio)
            
            with open(f"{args.data_dir}/transcripts.txt", 'w') as f:
                for i, text in enumerate(sample_texts):
                    f.write(f"audio_{i:04d}.wav|{text}\n")
        
        train_dataset = SpeechDataset(
            audio_dir=args.data_dir,
            transcript_file=os.path.join(args.data_dir, "transcripts.txt"),
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=text_tokenizer,
            max_audio_len=args.max_audio_len,
            max_text_len=args.max_text_len
        )
        val_dataset = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    else:
        val_loader = None
    
    # Create model
    from mamba_whisper import MambaWhisperASR
    
    model = MambaWhisperASR(
        vocab_size=text_tokenizer.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        n_mels=80,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss = evaluate(model, val_loader, device, text_tokenizer)
            print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.checkpoint_dir}/best_model.pt')
            print(f"Saved best model (loss: {best_loss:.4f})")
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{args.checkpoint_dir}/final_model.pt')
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-Whisper ASR")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/librispeech',
                        help='Directory containing audio files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--dataset', type=str, default='librispeech', choices=['librispeech', 'custom'],
                        help='Dataset to use')
    parser.add_argument('--train_limit', type=int, default=1000,
                        help='Limit training samples (for quick testing)')
    parser.add_argument('--val_limit', type=int, default=100,
                        help='Limit validation samples')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--d_state', type=int, default=16,
                        help='State dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=4,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4,
                        help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_audio_len', type=int, default=3000,
                        help='Max audio length')
    parser.add_argument('--max_text_len', type=int, default=200,
                        help='Max text length')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
