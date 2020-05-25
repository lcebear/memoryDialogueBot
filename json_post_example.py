import requests

while(1):
    try:
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        #TODO: Replace link with ngrok tunnel link, replace 1234 with actual userID and replace input_sentence with the user's question
        r = requests.post('http://127.0.0.1:5000/get_answer', json={"userID": 1234, "data": input_sentence})
        #print(r.status_code)
        print(r.json())
        #print(r)
        print(r.json()["answer"])
    except Exception as e:
        print(e)