from transformers import GPT2Tokenizer
from transformers import GPT2PreTrainedModel


a = GPT2Tokenizer.from_pretrained('gpt2') 
print(a.encode('king'))


b.encode('man')
a.encode('woman')

#add() method
a.add_tokens('king', 'queen')
