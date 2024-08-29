from transformers import BertTokenizer
a = BertTokenizer.from_pretrained('bert-base-uncased')
vocab = a.get_vocab()
print(vocab)
