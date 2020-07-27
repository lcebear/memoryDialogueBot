import agent_2 as agent


userID = "randomUser"
while True:
    try:
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        json_reply = agent.get_reply(input_sentence, userID, only_generated=False)
        if json_reply['error'] == None:
            pass 
            #json_reply['answer']
            #json_reply['component']
        else:
            print(json_reply['error'])
    except Exception as e:
        print("Exception in 'run_agent.py'",e)
