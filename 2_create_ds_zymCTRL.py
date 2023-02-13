import torch
import math
import pandas as pd
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
#from evaluate import logging
import pickle
import collections
import glob


def remove_characters(str, chars_list):
    for char in chars_list:
        if char == '<sep>':
            str = str.replace(char, ' ')
        else:
            str = str.replace(char, '')
    return str

def calculatePerplexity(input_ids,model,tokenizer):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(input_ids, model,special_tokens,device,tokenizer):
    input_ids = tokenizer.encode(input_ids,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
    	top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=600,
        eos_token_id=1,
        pad_token_id=0,
   	    do_sample=True,
   	    num_return_sequences=30)
    
    #Check sequence sanity, sequences not-truncated
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        pass
    #print("not enough sequences with short lengths!!")
        #generate and truncate:
        outputs = model.generate(
        input_ids, 
    	top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
   	    do_sample=True,
   	    num_return_sequences=50)
    
    ppls = [calculatePerplexity(output, model, tokenizer) for output in outputs ]
    
    inf_res = {}
    ppl = list(zip(ppls, [tokenizer.decode(x) for x in outputs]))
    ppl.sort(key=lambda i:i[0])
    ppl = list(set(ppl))
    first_seq,second_seq = ppl[:2]
    cond_tok = first_seq[1].split('<sep>')[0].replace(' ','')
    inf_res[cond_tok] = [(remove_characters(first_seq[1], special_tokens), first_seq[0]),(remove_characters(second_seq[1], special_tokens), second_seq[0])]
    return inf_res

if __name__=='__main__':
    device = torch.device("cuda")
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('/home/phil/Geraldene/transformers/examples/pytorch/language-modeling/ZymCTRL')
    model = GPT2LMHeadModel.from_pretrained('/home/phil/Geraldene/transformers/examples/pytorch/language-modeling/ZymCTRL').to(device) # change to new one
    df = pd.read_csv('random_validation_set.csv' )
    labels = df
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
    #print('Reading natural files...')
    #natural_files = glob.glob('/agh/projects/noelia/NLP/zymCTRL/sequences/natural_2/*.fasta')
    #labels = [i.split('/')[-1].split('_')[0] for i in natural_files]
    #print("we have :," len(labels), "labels")

    for label in tqdm(labels):
        print(label)
        for i in range(0,50):
            inf_res = main(label,model,special_tokens,device,tokenizer)
            for key,value in inf_res.items():
                for index, val in enumerate(value):
                    if os.path.exists(f"/home/phil/Geraldene/transformers/examples/pytorch/language-modeling/ZymCTRL/results_validation_random_results/set100_1/{label}_{i}_{index}.fasta"):
                        print(f"file {label} was already printed out")
                        continue
                    file_path = ('/home/phil/Geraldene/transformers/examples/pytorch/language-modeling/ZymCTRL/results_validation_random_results/set100_1/')
                    mode = "a" if os.path.exists(file_path) else "w"
                    fn = open(f"/home/phil/Geraldene/transformers/examples/pytorch/language-modeling/ZymCTRL/results_validation_random_results/set100_1/{label}_{i}_{index}.fasta", mode)
                    fn.write(f'>{val[1]}\n{label}_{i}_{index}\n{val[0]}')
                    fn.close()
