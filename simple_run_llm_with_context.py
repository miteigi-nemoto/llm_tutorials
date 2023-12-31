#Запускает работу с контекстом в float16
from transformers import pipeline,AutoTokenizer
import torch

checkpoint =  "OpenAssistant/llama2-13b-orca-8k-3319"

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
"""
[{'generated_text': "\nUse the following pieces of context to answer only on Russian language. If the answer is not contained in the context, please don't share false information.\n\nProbiotic products are considered functional foods because they benefit human health. \nQuestion: Сделай перевод с английского на русский язык. \n\nПробиотики считаются функциональными продуктами, потому что они способствуют здоровью человека."}]
"""
