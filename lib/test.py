import requests

def pratham(facts,text):
    url =  "http://localhost:8000/chat"

    data =  { 'facts' : facts,
                'text' : text 
                }
   
    response= requests.post(url, json=data)
    prediction=response.json()
    reply = prediction["reply"]
    return reply

if __name__=="__main__":
    user_input = input("Enter facts: ")

    user_list = user_input.split(',')
    facts= [element.strip() for element in user_list]
    text= str(input())
    i = pratham(facts,text)
    print(i)
