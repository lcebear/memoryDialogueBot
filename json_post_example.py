import requests

#different threads calling the api will create different threads in the api.
def call_api_question(userID, data):
    r = requests.post('http://127.0.0.1:5000/get_answer', json={"userID": userID, "data": data})
    return r
    
def call_api_self_disclosure(userID, data, topic):
    r = requests.post('http://127.0.0.1:5000/get_disclosure', json={"userID": userID, "data": data, "topic" :topic})
    return r
    
def call_api_self_disclosure_and_reflect(userID, data, topic):
    r = requests.post('http://127.0.0.1:5000/get_disclosure_and_reflect', json={"userID": userID, "data": data, "topic" :topic})
    return r
    
def call_api_get_question(userID, data):
    r = requests.post('http://127.0.0.1:5000/get_question', json={"userID": userID, "data": data})
    return r

options = [ "1. Ask question", "2. Self disclosure", "3. Self disclosure and reflect", "4. Get question"]
while(1):
    try:
        for opt in options:
            print(opt)
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        
        if input_sentence =="1":
            input_sentence = input('> ')
            r = call_api_question(1234, input_sentence)
            print(r.json())
            print(r.json()["answer"])
        if input_sentence =="2":
            input_sentence = input('> ')
            topic = input('>topic: ')
            r = call_api_self_disclosure(1234, input_sentence, topic)
            print(r.json())
            print(r.json()["answer"])
        if input_sentence =="3":
            input_sentence = input('> ')
            topic = input('>topic: ')
            r = call_api_self_disclosure_and_reflect(1234, input_sentence, topic)
            print(r.json())
            print(r.json()["answer"])
        if input_sentence == "4":
            input_sentence = input('> ')
            r = call_api_get_question(1234, input_sentence)
            print(r.json())
            print(r.json()["answer"])
    
    except Exception as e:
        print(e)