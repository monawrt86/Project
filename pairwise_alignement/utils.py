'''
Created on Jan 12, 2024

@author: danhbuithi
'''

import re 
import json 
import h5py 
import torch 
import numpy as np 
from typing import List 

from transformers import T5Tokenizer, T5EncoderModel


def pairwise_euclidean(x, y):
    z = x[:,:,None] - y[:, :, None].T
    return np.sqrt((z*z).sum(1))

def context_based_similarity(z):
            
    y = np.zeros(z.shape, dtype=float)
    y[:-1,:-1] = z[1:,1:]
    y[1:,1:] += z[:-1,:-1]
    
    f = np.zeros(z.shape, dtype=int) + 2
    f[0,:] = 1
    f[-1,:] = 1
    f[:,0] = 1
    f[:,-1] = 1
    y /= f 
    
    z = 0.6 * z + 0.4 * y
    
    return z 
    
def align_secondary_structure(aligned_seq, seq_structure):
    aligned_structure = []
    i = 0 
    for c in aligned_seq:
        if c == '-': 
            aligned_structure.append('-')
        else:
            aligned_structure.append(seq_structure[i])
            i += 1
    return ''.join(aligned_structure) 
        

def prottrans_T5_features(sequences: List[str], tokenizer: T5Tokenizer, llm: T5EncoderModel, device: str):
    '''
    Get amino acid-level features for sequences using ProtT5 protein language model
    '''
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
    # generate embeddings
    with torch.no_grad():
        embeddings = llm(input_ids=input_ids,attention_mask=attention_mask)
        embeddings = embeddings.last_hidden_state
    
    if device == 'cpu':
        return embeddings.detach().numpy()
    
    return embeddings.detach().cpu().numpy()


        
def list2hdf5(file_name, data_set):
    '''
    Save a dataset into hdf5 format
    '''
    file_writer = h5py.File(file_name, 'w')
    dt = h5py.string_dtype(encoding='ascii')
    dset = file_writer.create_dataset('data', (len(data_set),), dtype=dt, compression='gzip')
    for i, x in enumerate(data_set):
        dset[i] = json.dumps(x)
    file_writer.close()
    