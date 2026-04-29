import torch
import argparse
from assets import config, TOKENIZER_FILE
from lib import tokenizer_helper
from transformers import GPT2LMHeadModel, AutoTokenizer, PreTrainedTokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
parser.add_argument("-m", type=str)
parser.add_argument("--length", type=int, default=100)
args = parser.parse_args()

tokenizer = tokenizer_helper.try_load_tokenizer(TOKENIZER_FILE)
fast_tokenizer = tokenizer_helper.get_fast_tokenizer(tokenizer, config["tokenizer"]["eos_token"], config["tokenizer"]["unk_token"], config["tokenizer"]["pad_token"])

model = GPT2LMHeadModel.from_pretrained(args.m)
model.to("cuda") # type: ignore
model.eval()

tokenized_input = fast_tokenizer.encode(
  args.prompt,
  return_tensors="pt",
  padding=True,
  truncation=True
)
tokenized_input = tokenized_input.to("cuda")

attention_mask = (tokenized_input != fast_tokenizer.pad_token_id).long()
attention_mask = attention_mask.to("cuda")

with torch.no_grad():
  output = model.generate( # type: ignore
    tokenized_input,
    attention_mask=attention_mask,
    max_length=args.length,
    pad_token_id=fast_tokenizer.eos_token_id,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True, top_k=50,
    top_p=0.95, temperature=0.7
  )

output_text = fast_tokenizer.decode(output[0], skip_special_tokens=True)
print("\n" + "="*30)
print(output_text)
print("="*30)