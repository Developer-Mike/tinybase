import os
import re
import glob
import json
from lib import dataset_helper
from lib import tokenizer_helper
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_FILE, "r") as f:
  config = json.load(f)

OUTPUT_DIR = os.path.join(BASE_DIR, "out", config["version"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_FILE = os.path.join(OUTPUT_DIR, "tokenizer.json")
DATASET_DIR = dataset_helper.get_dataset_path(
  DATASETS_DIR,
  config["dataset"]["repo"],
  config["dataset"]["variant"],
  "train"
)
TOKENIZED_DATASET_DIR = dataset_helper.get_tokenized_dataset_path(
  DATASETS_DIR,
  config["dataset"]["repo"],
  config["dataset"]["variant"],
  "train"
)
checkpoints = sorted(
  glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
  key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0])
)
RESUME_CHECKPOINT = checkpoints[-1] if checkpoints else None

# Load model
model = GPT2LMHeadModel(GPT2Config(**config["model"]))
model.to("cuda") # type: ignore

parameter_count = sum(p.numel() for p in model.parameters())
print("Training model with the following configuration:")
print(f"\tModel version: {config['version']} ({parameter_count/1e6:.2f}M parameters)")
print(f"\tDataset config: {config['dataset']})")
print(f"\tTokenizer config: {config['tokenizer']}")
print(f"\tModel config: {config['model']}")
print(f"\tTraining config: {config['training']}")

# Load or train tokenizer
dataset = None
print("Loading tokenizer...")
tokenizer = tokenizer_helper.try_load_tokenizer(TOKENIZER_FILE)
if tokenizer is None:
  print("Loading dataset...")
  dataset = dataset_helper.load_dataset(
    DATASET_DIR,
    config["dataset"]["repo"],
    config["dataset"]["variant"],
    "train"
  )
  print("Dataset loaded")

  print("Training tokenizer...")
  tokenizer = tokenizer_helper.train_tokenizer(
    data=dataset[config["dataset"]["data_column"]],
    vocab_size=config["tokenizer"]["vocab_size"],
    unk_token=config["tokenizer"]["unk_token"]
  )
  tokenizer.save(TOKENIZER_FILE)
  print("Tokenizer trained")
print("Tokenizer loaded")

# Load or tokenize dataset
print("Loading tokenized dataset...")
tokenized_dataset = dataset_helper.try_load_tokenized_dataset(TOKENIZED_DATASET_DIR)
if tokenized_dataset is None:
  print("Loading dataset...")
  dataset = dataset or dataset_helper.load_dataset(
    DATASETS_DIR,
    config["dataset"]["repo"],
    config["dataset"]["variant"],
    "train"
  )
  print("Dataset loaded")

  print("Tokenizing dataset...")
  tokenized_dataset = dataset_helper.tokenize_dataset(tokenizer, config["dataset"]["data_column"], dataset)
  dense_tokenized_dataset = dataset_helper.densify_tokenized_dataset(tokenized_dataset, config["model"]["n_positions"])
  dense_tokenized_dataset.save_to_disk(TOKENIZED_DATASET_DIR)
  print("Dataset tokenized")
print("Tokenized dataset loaded")

fast_tokenizer = PreTrainedTokenizerFast(
  tokenizer_object=tokenizer,
  clean_up_tokenization_spaces=True
)
fast_tokenizer.add_special_tokens({"pad_token": config["tokenizer"]["pad_token"]})

training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  per_device_train_batch_size=8,
  logging_steps=100, save_steps=1000,
  fp16=True, **config["training"]
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset,
  data_collator=DataCollatorForLanguageModeling(
    tokenizer=fast_tokenizer,
    mlm=False
  )
)

print(f"{'Resuming' if RESUME_CHECKPOINT else 'Starting'} training...")
trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
