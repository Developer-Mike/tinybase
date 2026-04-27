import os
import json

def get_valid_dirname(name: str) -> str:
  return "".join([c.lower() for c in name if c.isalnum()])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_FILE, "r") as f:
  config = json.load(f)

OUTPUT_DIR = os.path.join(BASE_DIR, "out", config["version"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_FILE = os.path.join(OUTPUT_DIR, "tokenizer.json")
DATASET_DIR = os.path.join(
  DATASETS_DIR, 
  get_valid_dirname(config["dataset"]["repo"]), 
  get_valid_dirname(config["dataset"]["variant"]), 
  "train", 
  "raw"
)
TOKENIZED_DATASET_DIR = os.path.join(
  DATASETS_DIR, 
  get_valid_dirname(config["dataset"]["repo"]), 
  get_valid_dirname(config["dataset"]["variant"]), 
  "train", 
  "tokenized"
)