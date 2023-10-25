#Запускает работу с контекстом
from transformers import pipeline,AutoTokenizer
import torch

checkpoint =  "Open-Orca/Mistral-7B-OpenOrca"

model_kwargs = {"device_map": "auto", "load_in_8bit": False}
q_tokenizer = AutoTokenizer.from_pretrained(checkpoint )
 
p = pipeline("text-generation", checkpoint,  torch_dtype=torch.float16,   tokenizer=q_tokenizer, past_key_values=None, batch_size=512, max_new_tokens=500,temperature=0 ,  model_kwargs=model_kwargs   ) 

context = """
Use the following pieces of context to answer only on Russian language. If the answer is not contained in the context, please don't share false information.

Probiotic products are considered functional foods because they benefit human health. 
Question: Сделай перевод с английского на русский язык. 
"""

result = p(context )
print(result)
