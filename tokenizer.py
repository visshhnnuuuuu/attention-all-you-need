from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models
from tokenizers import trainers
from tokenizers import pre_tokenizers
from tokenizers import decoders


class Tokenizer:

    def __init__(
        self,
        data,
        language,
        vocab_size=8000
    ):

        self.language = language

        # -----------------------------------------
        # Special tokens
        # -----------------------------------------

        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3

        special_tokens = [
            "<PAD>",
            "<SOS>",
            "<EOS>",
            "<UNK>"
        ]

        # -----------------------------------------
        # Create BPE tokenizer
        # -----------------------------------------

        self.tokenizer = HFTokenizer(
            models.BPE(
                unk_token="<UNK>"
            )
        )

        # Whitespace + punctuation aware
        self.tokenizer.pre_tokenizer = (
            pre_tokenizers.ByteLevel(
                add_prefix_space=False
            )
        )

        # -----------------------------------------
        # Trainer
        # -----------------------------------------

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=2
        )

        # -----------------------------------------
        # Training sentences
        # -----------------------------------------

        sentences = []

        for item in data:

            sentence = item[language].strip()

            if sentence:

                sentences.append(sentence)

        # -----------------------------------------
        # Train tokenizer
        # -----------------------------------------

        self.tokenizer.train_from_iterator(
            sentences,
            trainer=trainer
        )

        # -----------------------------------------
        # Decoder
        # -----------------------------------------

        self.tokenizer.decoder = (
            decoders.ByteLevel()
        )

        # -----------------------------------------
        # Vocabulary
        # -----------------------------------------

        self.vocab_size = (
            self.tokenizer.get_vocab_size()
        )

    def encode_text(self, text):

        encoding = self.tokenizer.encode(
            text
        )

        return encoding.ids

    def decode_tokens(self, tokens):

        if not tokens:

            return ""

        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True
        )