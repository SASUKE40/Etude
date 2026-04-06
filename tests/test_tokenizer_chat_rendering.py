from dataclasses import dataclass

from etude.tokenizer import HuggingFaceTokenizer


@dataclass
class MockEncoding:
    ids: list[int]


@dataclass
class MockAddedToken:
    content: str


class MockHFBackend:
    def __init__(self, special_tokens):
        self.special_tokens = special_tokens
        self.reverse_special = {value: key for key, value in special_tokens.items()}

    def get_vocab_size(self):
        return max(self.reverse_special.keys(), default=0) + 1

    def get_added_tokens_decoder(self):
        return {token_id: MockAddedToken(token) for token, token_id in self.special_tokens.items()}

    def id_to_token(self, token_id):
        return self.reverse_special.get(token_id, chr(token_id))

    def token_to_id(self, text):
        return self.special_tokens.get(text)

    def encode(self, text, add_special_tokens=False):
        return MockEncoding([ord(char) for char in text])

    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for token_id in ids:
            if token_id in self.reverse_special:
                if not skip_special_tokens:
                    pieces.append(self.reverse_special[token_id])
            else:
                pieces.append(chr(token_id))
        return "".join(pieces)

    def save(self, path):
        raise NotImplementedError


def test_hf_tokenizer_renders_qwen_conversation_and_completion_prompt():
    tokenizer = HuggingFaceTokenizer(
        MockHFBackend(
            {
                "<|endoftext|>": 1000,
                "<|im_start|>": 1001,
                "<|im_end|>": 1002,
            }
        )
    )
    conversation = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)
    rendered = tokenizer.decode(ids)

    assert rendered == (
        "<|endoftext|><|im_start|>user\nYou are helpful.\n\nSay hi.<|im_end|>\n"
        "<|im_start|>assistant\nHi there!<|im_end|>\n"
    )
    assert len(ids) == len(mask)
    assert sum(mask) == len("Hi there!") + 2  # assistant content + <|im_end|> + newline

    prompt_ids = tokenizer.render_for_completion(conversation)
    prompt = tokenizer.decode(prompt_ids)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert prompt == (
        "<|endoftext|><|im_start|>user\nYou are helpful.\n\nSay hi.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_hf_tokenizer_renders_legacy_chat_tokens():
    tokenizer = HuggingFaceTokenizer(
        MockHFBackend(
            {
                "<|bos|>": 1000,
                "<|user_start|>": 1001,
                "<|user_end|>": 1002,
                "<|assistant_start|>": 1003,
                "<|assistant_end|>": 1004,
                "<|python_start|>": 1005,
                "<|python_end|>": 1006,
                "<|output_start|>": 1007,
                "<|output_end|>": 1008,
            }
        )
    )
    conversation = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)
    rendered = tokenizer.decode(ids)

    assert rendered == "<|bos|><|user_start|>Hello<|user_end|><|assistant_start|>World<|assistant_end|>"
    assert len(ids) == len(mask)
    assert sum(mask) == len("World") + 1  # assistant content + <|assistant_end|>
