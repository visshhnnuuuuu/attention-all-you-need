import tiktoken
class Tokenizer:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def encode_text(self, text):
        tokens = self.encoding.encode(text)
        return tokens

    def decode_tokens(self, tokens):
        text = self.encoding.decode(tokens)
        return text

    





