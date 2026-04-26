import os
import tokenizers

def try_load_tokenizer(dir: str) -> tokenizers.Tokenizer | None:
  if os.path.exists(dir):
    return tokenizers.Tokenizer.from_file(dir)
  return None

def train_tokenizer(data: list[str], vocab_size: int, unk_token: str) -> tokenizers.Tokenizer:
  tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=unk_token))
  tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)

  trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=[unk_token]
  )

  tokenizer.train_from_iterator(data, trainer=trainer)
  return tokenizer