import retrieval_component as retrieval

input_sentence = ''
while(1):
    
    user_id = 1337
    user_msg_history = []
    
    #fetch the current user's entry from the dataframe if exist or add new entry
    curr_user = retrieval.user_history.loc[retrieval.user_history['userID'] == user_id]
    if len(curr_user) > 0:
        if len(curr_user.message_history.iloc[0]) > 0:
            user_msg_history = curr_user.message_history.iloc[0]        
    else:
        retrieval.user_history = retrieval.user_history.append({'userID' : user_id , 'message_history' : [], 'true_sentiment' : [] } , ignore_index=True)
        print("Added ", user_id, "to user history")
        curr_user = retrieval.user_history.loc[retrieval.user_history['userID'] == user_id]

    try:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
            
        answer, sim_val_2, answer_id_2, max_sim_q_2 = retrieval.find_question_n_answer_retrieval(input_sentence)
        
        if answer != None:
            previous_answer = retrieval.check_msg_history(user_msg_history, answer_id_2)
            if previous_answer != None:
                answer = previous_answer
            else:
                if answer_id_2 in retrieval.exception_qid:
                    pass
                else:
                    user_msg_history.append((input_sentence, answer, answer_id_2))
                    #Todo, implement true_sentiment
                    retrieval.user_history.at[curr_user.index.values[0], 'message_history'] = user_msg_history
        print(max_sim_q_2, sim_val_2, answer_id_2, "\n")
        print(answer)
        
    except KeyError:
        print("Error: Encountered unknown word.")

