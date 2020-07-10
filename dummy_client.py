#Purpose: Testing generative model, concurrent user calls 

import requests
from timeit import default_timer as timer
from threading import Thread

def call_api(userID, data):
    r = requests.post('http://127.0.0.1:5000/get_answer', json={"userID": userID, "data": data})
    return r

counter = 0
userID = "main_dummy"
while(1):
    counter +=1
    #input_sentence = input('> ')
    # Check if it is quit case
    #if input_sentence == 'q' or input_sentence == 'quit': break
    
    
    data ="Have you seen number " + str(counter) + "?"
    start = timer()
    Thread(target=call_api, args=("dummy",data)).start()
    Thread(target=call_api, args=("dummy",data)).start()
    r = call_api(userID,data)
    ender = timer()
    print(ender-start)
    print(r.json())
    

        
        
