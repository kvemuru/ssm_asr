# Mamba-Whisper ASR

A version of a State Space Model (SSM) for Automatic Speech Recognition, inspired by Mamba and Whisper architectures. This model is work in progress and not yet ready to train/test with a benchmark dataset. 

## Overview

This project implements a simplified State Space Model for speech recognition, combining:
- **Mamba-style SSM blocks** - Efficient sequence modeling with selective state spaces
- **Whisper-style architecture** - Encoder-decoder for audio-to-text generation

## Model Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Input    │────▶│   Audio Encoder  │────▶│  SSM Encoder   │
│  (Mel Spec)     │     │  (Conv + Pos Em) │     │  (4 layers)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Text Output   │◀────│  Text Decoder   │◀────│  SSM Decoder   │
│  (Transcription)│     │ (Cross-Attn)    │     │  (4 layers)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Components

1. **Audio Encoder**: Converts mel spectrograms to embeddings using 1D convolutions
2. **SSM Encoder**: Stack of SSM blocks with self-attention
3. **SSM Decoder**: Stack of SSM blocks with cross-attention to encoder output
4. **Text Tokenizer**: Character-level tokenization

## Installation

### 1. Create conda environment

```bash
conda create -n ssm_asr python=3.12 -y
conda activate ssm_asr
```

### 2. Install dependencies

```bash
# PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Audio processing
pip install librosa soundfile scipy

# Training utilities
pip install tqdm numpy
```

## Quick Start

### Test the model

```bash
cd /mnt/c/Users/kvemu/Projects/ssm_asr
python mamba_whisper.py
```

### Train with custom dataset

```bash
# Prepare your data
# - Put audio files (.wav) in ./data/speech/
# - Create transcripts.txt with format: filename.wav|text

python train.py --dataset custom --data_dir ./data/speech --epochs 10
```

### Train with LibriSpeech

```bash
# Downloads automatically (~50GB for full dataset)
python train.py --dataset librispeech --epochs 10 --batch_size 4
```

## Training Options

### Data Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset type | `librispeech` |
| `--data_dir` | Data directory | `./data/librispeech` |
| `--checkpoint_dir` | Save checkpoints | `./checkpoints` |
| `--train_limit` | Limit training samples | 1000 |
| `--val_limit` | Limit validation samples | 100 |

### Model Options

| Option | Description | Default |
|--------|-------------|---------|
| `--d_model` | Model dimension | 256 |
| `--d_state` | SSM state dimension | 16 |
| `--num_encoder_layers` | Encoder layers | 4 |
| `--num_decoder_layers` | Decoder layers | 4 |
| `--num_heads` | Attention heads | 4 |
| `--dropout` | Dropout rate | 0.1 |
| `--max_audio_len` | Max audio frames | 3000 |
| `--max_text_len` | Max text tokens | 200 |

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch_size` | Batch size | 4 |
| `--epochs` | Number of epochs | 10 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--weight_decay` | Weight decay | 0.01 |

## Dataset Format

### Custom Dataset

```
data/
├── audio_001.wav
├── audio_002.wav
└── transcripts.txt
```

**transcripts.txt format:**
```
audio_001.wav|hello world
audio_002.wav|this is a test
```

### LibriSpeech

The dataset is automatically downloaded from [OpenSLR](https://openslr.org/):

- **train-clean-100**: ~100 hours of clean speech
- **train-clean-360**: ~360 hours of clean speech  
- **dev-clean**: ~5 hours for validation
- **test-clean**: ~5 hours for testing

To download manually:
```bash
# Download from https://openslr.org/12/
# - train-clean-100.tar.gz (~6GB)
# - dev-clean.tar.gz (~400MB)
```

## Checkpoints

Checkpoints are saved in `./checkpoints/`:

```
checkpoints/
├── best_model.pt    # Best model based on training loss
└── final_model.pt  # Final model after training
```

## Inference

### Using trained model

```python
import torch
from mamba_whisper import MambaWhisperASR, AudioTokenizer

# Load model
model = MambaWhisperASR(vocab_size=34, d_model=256, ...)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Prepare audio
audio_tokenizer = AudioTokenizer()
mel_spec = audio_tokenizer(audio).unsqueeze(0)

# Transcribe
transcription = model.transcribe(mel_spec, max_length=100)
```

## Performance

- **Parameters**: ~11M (with default settings)
- **Training**: Supports GPU (CUDA) and CPU
- **Memory**: Adjust `batch_size` and `max_audio_len` for memory constraints

## Project Structure

```
ssm_asr/
├── mamba_whisper.py     # Model architecture
├── train.py             # Training script
├── README.md            # This file
├── data/                # Dataset directory
└── checkpoints/         # Saved models
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- librosa
- numpy
- tqdm

## License

MIT License

## References

- [Mamba: Linear-time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [Samba-ASR: State-of-the-Art Speech Recognition with SSMs](https://arxiv.org/abs/2501.02832)
