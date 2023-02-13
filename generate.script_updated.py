import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from evaluate import logging
import pickle
import collections
import glob
import math

def remove_characters(sequence, char_list):
    "removing characters"
    columns = sequence.split('<sep>')
    try:
        seq = columns[1]
    except Exception as e:
        print("no separator was found in the sequence","sequence is", sequence)
        print(e)
        return ''
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model,special_tokens,device,tokenizer):
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
    	top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
   	    do_sample=True,
   	    num_return_sequences=20)
    
    # Check sequence sanity, sequences not-truncated
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")
    
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]
    ppls.sort(key=lambda i:i[1])
    ppls = list(set(ppls))
    inf_res={}
    inf_res[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return inf_res

if __name__=='__main__':
    device = torch.device("cuda:2")
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('/home/ec2-user/SageMaker/Zymctrl/')
    model = GPT2LMHeadModel.from_pretrained('/home/ec2-user/SageMaker/Zymctrl/').to(device) # change to new one
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
    labels=['2.5.1.63']
    print(len(labels))

    for label in tqdm(labels):
        print(label)
        for i in range(2126,60000):
            inf_res = main(label,model,special_tokens,device,tokenizer)
            for key,value in inf_res.items():
                for index, val in enumerate(value):            
                    fn = open(f"/home/ec2-user/SageMaker/Zymctrl/results/fluorinase_zeroshot/{label}_{i}_{index}.fasta", "w")
                    fn.write(f'>{label}_{i}_{index}\t{val[1]}\n{val[0]}')
                    fn.close()

