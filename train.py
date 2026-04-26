import os
import re
import glob
import json
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_FILE, "r") as f:
  config = json.load(f)

OUTPUT_DIR = os.path.join(BASE_DIR, "out", config["version"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
checkpoints = sorted(
  glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
  key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0])
)
RESUME_CHECKPOINT = checkpoints[-1] if checkpoints else None

model = GPT2LMHeadModel(GPT2Config(**config["model"]))
model.to("cuda") # type: ignore

parameter_count = sum(p.numel() for p in model.parameters())
print(f"{'Starting' if not RESUME_CHECKPOINT else 'Continuing'} training model")
print(f"\tModel version: {config['version']} ({parameter_count/1e6:.2f}M parameters)")
print(f"\tModel config: {config['model']}")
print(f"\tTraining config: {config['training']}")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
tokenized_dataset = dataset.map(
  lambda x: tokenizer(x["text"], truncation=True, max_length=512),
  remove_columns=dataset.column_names,
  batched=True, num_proc=os.cpu_count()
)

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
  data_collator=data_collator
)

trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
