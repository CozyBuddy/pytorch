from huggingface_hub import login

api_token = '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'
login(api_token)

import torch
import wandb
from sklearn.model_selection import train_test_split

from transformers import AutoModelForCausalLM , AutoTokenizer ,TrainingArguments, pipeline ,Trainer

from transformers.integrations import WandbCallback
from trl import SFTTrainer
import evaluate

model_name = 'google/gemma-3-4b-it'
model = AutoModelForCausalLM.from_pretrained(model_name ,device_map='auto' , dtype=torch.bfloat16 , attn_implementation='eager')
tokenizer = AutoTokenizer.from_pretrained(model_name)

import datasets
dataset = datasets.load_dataset('jaehy12/news3')
element = dataset['train'][1]
print(element)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
input_text = ''
def change_inference_chat_format(input_text):
    return [
        {'role' : 'user' , 'content' : f'{input_text}'},
        {'role' : 'assistant' , 'content' : """""" },
        {'role' : 'user' , 'content' :'중요한 키워드 5개를 뽑아주세요.'},
        {'role' : 'assistant' , 'content' : ""}
    ]

prompt = change_inference_chat_format(input_text)

inputs = tokenizer.apply_chat_template(prompt , tokenize=True ,add_generation_prompt=True , return_tensors='pt' ).to(device)
outputs = model.generate(input_ids=inputs.to(device) ,max_new_tokens=256)
print(tokenizer.decode(outputs[0] , skip_special_tokens=True))

def change_inference_chat_format2(input_text):
    return [
        {'role' : 'user' , 'content' : f'{input_text}'},
        {'role' : 'assistant' , 'content' : "한국어 요약 : \n" },

    ]

prompt = change_inference_chat_format2(input_text)

inputs = tokenizer.apply_chat_template(prompt , tokenize=True ,add_generation_prompt=True , return_tensors='pt').to(device)

