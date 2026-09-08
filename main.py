import torch

from tokenizer import Tokenizer
from embedding import Embedding
from positionalencoding import PositionalEncoding
from dataset import create_dataloader
from transformer import Transformer
from train import train
from inference import generate
from data_loader import load_translation_data


# ============================================
# 1. Load Real Dataset
# ============================================

data = load_translation_data(
    num_samples=50000
)

print(
    f"Total dataset pairs: {len(data)}"
)


# ============================================
# 2. Train / Validation Split
# ============================================

split_index = int(
    0.9 * len(data)
)

train_data = data[:split_index]

validation_data = data[split_index:]

print(
    f"Training pairs: {len(train_data)}"
)

print(
    f"Validation pairs: {len(validation_data)}"
)


# ============================================
# 3. Create BPE Tokenizers
# ============================================

src_tokenizer = Tokenizer(
    train_data,
    "src",
    vocab_size=8000
)

tgt_tokenizer = Tokenizer(
    train_data,
    "tgt",
    vocab_size=8000
)

print(
    f"Source vocabulary size: "
    f"{src_tokenizer.vocab_size}"
)

print(
    f"Target vocabulary size: "
    f"{tgt_tokenizer.vocab_size}"
)


# ============================================
# 4. Configuration
# ============================================

d_model = 128

num_heads = 8

d_ff = 512

num_layers = 2

max_len = 48

batch_size = 32

dropout = 0.1

# ============================================
# 5. Create Training DataLoader
# ============================================

train_loader = create_dataloader(
    data=train_data,
    src_tokenizer=src_tokenizer,
    tgt_tokenizer=tgt_tokenizer,
    max_len=max_len,
    batch_size=batch_size,
    shuffle=True
)


# ============================================
# 6. Create Validation DataLoader
# ============================================

validation_loader = create_dataloader(
    data=validation_data,
    src_tokenizer=src_tokenizer,
    tgt_tokenizer=tgt_tokenizer,
    max_len=max_len,
    batch_size=batch_size,
    shuffle=False
)


# ============================================
# 7. Create Source Embedding
# ============================================

src_embedding = Embedding(
    vocab_size=src_tokenizer.vocab_size,
    embedding_dim=d_model
)


# ============================================
# 8. Create Target Embedding
# ============================================

tgt_embedding = Embedding(
    vocab_size=tgt_tokenizer.vocab_size,
    embedding_dim=d_model
)


# ============================================
# 9. Create Positional Encoding
# ============================================

src_positional_encoding = PositionalEncoding(
    d_model=d_model,
    max_len=max_len
)

tgt_positional_encoding = PositionalEncoding(
    d_model=d_model,
    max_len=max_len
)


# ============================================
# 10. Create Transformer
# ============================================

model = Transformer(
    src_embedding=src_embedding,
    tgt_embedding=tgt_embedding,
    src_positional_encoding=src_positional_encoding,
    tgt_positional_encoding=tgt_positional_encoding,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    num_layers=num_layers,
    dropout=dropout
)

# ============================================
# 11. Train Model
# ============================================

train(
    model=model,
    dataloader=train_loader,
    validation_loader=validation_loader,
    pad_token_id=tgt_tokenizer.pad_token,
    src_pad_token_id=src_tokenizer.pad_token,
    num_epochs=10,
    learning_rate=1e-3,
    device="cpu"
)


# ============================================
# 12. Test Translation
# ============================================

test_sentences = [

    "मैं एक सेब खाता हूँ",

    "मुझे किताबें पसंद हैं",

    "मैं स्कूल जाता हूँ",

    "वह पानी पीती है",

    "यह एक घर है"

]


print(
    "\n========== TRANSLATION TEST ==========\n"
)


for sentence in test_sentences:

    translation = generate(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_text=sentence,
        max_len=max_len,
        device="cpu"
    )

    print(
        "Hindi   :",
        sentence
    )

    print(
        "English :",
        translation
    )

    print()