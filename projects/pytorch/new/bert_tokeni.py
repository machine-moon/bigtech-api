from transformers import BertTokenizer
from transformers import BertPreTrainedModel
from transformers import B

a= BertTokenizer.lo
def play():
    #ask for user input
    #give options, encode or decode or add() subtract()
    print("1. Encode")
    print("2. Decode")
    print("3. Add")
    print("4. Subtract")
    #get user input
    choice = input("Enter your choice: ")
    #perform the operation
    if choice == '1':
        print(encode())
    elif choice == '2':
        print(decode())
    elif choice == '3':
        print(add())
    elif choice == '4':
        print(subtract())
    else:
        print("Invalid choice")

def encode():
    #ask for input
    user_input = input("Enter something to encode: ")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_encode = tokenizer.encode(user_input)
    return tokens_encode
    
def decode():
    #ask for input ( should be a list [101, 102, 103])
    as_list = input("Enter a list to decode: ")    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_decode = tokenizer.decode(as_list)
    return tokens_decode

def addition():
    input1 = input("input1: ")
    #first encode the input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_encode = tokenizer.encode(input1)
    #ask the user for a second input
    input2 = input("input2: ")
    #encode the second input
    tokens_encode2 = tokenizer.encode(input2)
    #add the two encodings
    #add as in one[0]+two[0], one[1]+two[1]...
    tokens_add = [x + y for x, y in zip(tokens_encode, tokens_encode2)]    
    #decode the result
    tokens_decode = tokenizer.decode(tokens_add)
    return tokens_decode

def subtract():
    input1 = input("input1: ")
    #first encode the input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_encode = tokenizer.encode(input1)
    #ask the user for a second input
    input2 = input("input2: ")
    #encode the second input
    tokens_encode2 = tokenizer.encode(input2)
    #add the two encodings
    #add as in one[0]-two[0], one[1]-two[1]...
    tokens_subtract = [x - y for x, y in zip(tokens_encode, tokens_encode2)]
    #decode the result
    tokens_decode = tokenizer.decode(tokens_subtract)
    return tokens_decode




def add(input1, input2):
    #input1 = input("input1: ")
    #first encode the input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_encode = tokenizer.encode(input1)
    #ask the user for a second input
    #input2 = input("input2: ")
    #encode the second input
    tokens_encode2 = tokenizer.encode(input2)
    #add the two encodings
    #add as in one[0]+two[0], one[1]+two[1]...
    tokens_add = [x + y for x, y in zip(tokens_encode, tokens_encode2)]    
    #decode the result
    tokens_decode = tokenizer.decode(tokens_add)
    return tokens_decode

def sub(input1, input2):
    #input1 = input("input1: ")
    #first encode the input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_encode = tokenizer.encode(input1)
    #ask the user for a second input
    #input2 = input("input2: ")
    #encode the second input
    tokens_encode2 = tokenizer.encode(input2)
    #add the two encodings
    #add as in one[0]-two[0], one[1]-two[1]...
    tokens_subtract = [x - y for x, y in zip(tokens_encode, tokens_encode2)]
    #decode the result
    tokens_decode = tokenizer.decode(tokens_subtract)
    return tokens_decode




def encode():
    #ask for input
    user_input = input("Enter something to encode: ")
    
    d="queen"
    test = BertTokenizer.from_pretrained('bert-base-uncased')
    c = test.encode(d)
    return tokens_encode
    
    
    
