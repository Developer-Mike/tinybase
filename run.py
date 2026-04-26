import torch
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
parser.add_argument("-m", type=str)
parser.add_argument("--length", type=int, default=100)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(args.model_path)
model.to("cuda") # type: ignore
model.eval()

tokenized_input = tokenizer.encode(args.prompt, return_tensors="pt")
tokenized_input.to("cuda")

with torch.no_grad():
  output = model.generate( # type: ignore
    tokenized_input,
    max_length=args.length,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True, top_k=50,
    top_p=0.95, temperature=0.7
  )

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n" + "="*30)
print(output_text)
print("="*30)