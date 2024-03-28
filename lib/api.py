import uvicorn 
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

model_dir= "F:\golang-LCO\model\output"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

if torch.cuda.is_available():
    model = model.cuda()

user_input = input("Enter facts: ")

user_list = user_input.split(',')
facts= [element.strip() for element in user_list]    


flatten = lambda l: [item for sublist in l for item in sublist]

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=tokenizer.eos_token_id):
    full_msg = model.generate(bot_input_ids, do_sample=True,
                                              top_k=10, top_p=92, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg


def predict(fact, text):

    personas = []

    for n in fact:
        personas.append(n + tokenizer.eos_token)
    personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))



    dialog_hx = []
  
    user_inp = tokenizer.encode(text + tokenizer.eos_token)
    dialog_hx.append(user_inp)
            
    bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()
    msg = generate_next(bot_input_ids)
    dialog_hx.append(msg)

    
    bot_response =tokenizer.decode(msg, skip_special_tokens=True)
    print(bot_response)

    return bot_response



class InputData(BaseModel):
    text: str 

@app.get('/')
def index():
    return { 'message' : 'Hello, stranger'}    


@app.post('/chat')
def chat(data: InputData):
    

    text= data.text
    reply = predict(facts, text)
    return { 
        'reply' : reply 
        }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)    



