# load the protein language model
import torch
from transformers import T5Tokenizer, T5EncoderModel

print (torch.__version__)

device = 'cpu'
prot_model_name = 'Rostlab/prot_t5_xl_uniref50'
tokenizer = T5Tokenizer.from_pretrained(prot_model_name, legacy=True)
print ("tokenizer finish")
model = T5EncoderModel.from_pretrained(prot_model_name).to(device) 
print ("succes")