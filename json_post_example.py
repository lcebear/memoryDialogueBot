import requests
import concurrent.futures


#Todo peilin needs to have different threads that call the api, perhaps his threads are already created by rasa.
#different threads calling the api will create different threads in the api.
def call_api(userID, data):
    r = requests.post('http://127.0.0.1:5000/get_answer', json={"userID": userID, "data": data})
    return r
    


while(1):
    try:
        
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        
        
        #with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            #future = executor.submit(call_api, 56,input_sentence)
            ##return_value = future.result()
            #print(return_value.json()["answer"])
        
        #threads = Thread(target=call_api, args=(56,input_sentence, results))
        #threads.start()
        
        #TODO: Replace link with ngrok tunnel link, replace 1234 with actual userID and replace input_sentence with the user's question
        r = call_api(1234, input_sentence)
        #print(r.status_code)
        print(r.json())
        #print(r)
        print(r.json()["answer"])
        #threads.join()
        #print(results)
        
        
    except Exception as e:
        print(e)