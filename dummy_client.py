#Purpose: Testing generative model, concurrent user calls 

import requests
from timeit import default_timer as timer

def call_api(userID, data):
    r = requests.post('http://127.0.0.1:5000/get_answer', json={"userID": userID, "data": data})
    return r

counter = 0
userID = "dummy"
while(1):
    counter +=1
    #input_sentence = input('> ')
    # Check if it is quit case
    #if input_sentence == 'q' or input_sentence == 'quit': break
    
    
    data ="Have you seen number " + str(counter) + "?"
    start = timer()
    r = call_api(userID,data)
    ender = timer()
    print(ender-start)
    print(r.json())
    

        
        
