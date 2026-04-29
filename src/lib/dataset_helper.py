import os
import datasets
import tokenizers
from typing import cast

def load_dataset(dir: str, repo: str, variant: str, split: str) -> datasets.Dataset:
  return datasets.load_dataset(repo, variant, split=split, cache_dir=dir)

def try_load_tokenized_dataset(dir: str) -> datasets.Dataset | None:
  if os.path.exists(dir):
    return cast(datasets.Dataset, datasets.load_from_disk(dir))
  return None

def densify_tokenized_dataset(dataset: datasets.Dataset, block_size: int) -> datasets.Dataset:
  def group_texts(examples):
    concatenated = []
    for ids in examples["input_ids"]:
      concatenated.extend(ids)

    total_length = len(concatenated)
    total_length = (total_length // block_size) * block_size

    result = {
      "input_ids": [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
      ]
    }

    result["labels"] = result["input_ids"].copy()
    return result

  return dataset.map(
    group_texts,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count() - 1
  )

def tokenize_dataset(tokenizer: tokenizers.Tokenizer, data_column: str, dataset) -> datasets.Dataset:
  def batch_tokenize(batch):
    encodings = tokenizer.encode_batch(
      batch[data_column],
      add_special_tokens=False
    )

    return {"input_ids": [e.ids for e in encodings]}

  tokenized_dataset = dataset.map(
    batch_tokenize,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count() - 1
  )

  return tokenized_dataset
