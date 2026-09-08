import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):

    def __init__(
        self,
        data,
        src_tokenizer,
        tgt_tokenizer,
        max_len
    ):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        # ==========================================
        # 1. Get sentences
        # ==========================================

        src_text = self.data[index]["src"]
        tgt_text = self.data[index]["tgt"]


        # ==========================================
        # 2. Tokenize separately
        # ==========================================

        src_tokens = self.src_tokenizer.encode_text(
            src_text
        )

        tgt_tokens = self.tgt_tokenizer.encode_text(
            tgt_text
        )


        # ==========================================
        # 3. Add SOS and EOS
        # ==========================================

        src_tokens = (
            [self.src_tokenizer.sos_token]
            + src_tokens
            + [self.src_tokenizer.eos_token]
        )

        tgt_tokens = (
            [self.tgt_tokenizer.sos_token]
            + tgt_tokens
            + [self.tgt_tokenizer.eos_token]
        )


        # ==========================================
        # 4. Truncate
        # ==========================================

        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len]


        # ==========================================
        # 5. Padding
        # ==========================================

        src_tokens += [
            self.src_tokenizer.pad_token
        ] * (
            self.max_len - len(src_tokens)
        )

        tgt_tokens += [
            self.tgt_tokenizer.pad_token
        ] * (
            self.max_len - len(tgt_tokens)
        )


        # ==========================================
        # 6. Convert to tensors
        # ==========================================

        src_tokens = torch.tensor(
            src_tokens,
            dtype=torch.long
        )

        tgt_tokens = torch.tensor(
            tgt_tokens,
            dtype=torch.long
        )


        return src_tokens, tgt_tokens


def create_dataloader(
    data,
    src_tokenizer,
    tgt_tokenizer,
    max_len,
    batch_size=32,
    shuffle=True
):

    dataset = TranslationDataset(
        data=data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader