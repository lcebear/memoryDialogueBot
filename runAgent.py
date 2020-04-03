import agent

input_sentence = ''
similarity_threshhold = 0.8 #Below this threshhold, the question is generated isntead of retrieved.
def get_reply(input_sentence):
    try:
        db_name = "likes"
        using_generated = False
        processed_text_input, user_noun, orig_noun, noun_topics  = agent.process_user_input(input_sentence)
        answer_id, max_sim_val, max_sim_q = agent.find_question_template(processed_text_input)
        
        #Check the retrieval structure if a higher question similarity is found.
        answer, sim_val_2, answer_id_2, max_sim_q_2 = agent.find_question_n_answer_retrieval(input_sentence)
        print(sim_val_2, max_sim_val)
        print(input_sentence)
        if (sim_val_2 < similarity_threshhold) and (max_sim_val < similarity_threshhold):
            answer = generated_reply_helper(input_sentence)
            using_generated = True
        else:
            if sim_val_2 > max_sim_val:
                if answer != None:
                    #print(answer)
                    max_sim_val = sim_val_2
                    answer_id = answer_id_2
                    max_sim_q = max_sim_q_2
                    db_name = "retrieval"
                    
                else:
                    answer = generated_reply_helper(input_sentence)
                    using_generated = True
            #print(answer_id)
            elif user_noun != None:
                #print(user_noun, orig_noun)
                answer, nouns, answer_sentiment = agent.fetch_answer_template(answer_id, user_noun, noun_topics)
                #print('\n',input_sentence)#, question_sentiment)
                answer = answer.sample().iloc[0]
                answer = agent.process_agent_output(answer,
                                                 user_noun, nouns,noun_topics, answer_sentiment)
            #The likes/dislikes template cannot handle the user input, introduce generative model.    
            else:
                answer = generated_reply_helper(input_sentence)
                using_generated = True
        print(answer, max_sim_q, answer_id, max_sim_val)
        ret_dict = { 'answer' : answer, 'matched_question' : max_sim_q, 'answer_id' : answer_id, 'similarity' : max_sim_val,
        'processed_input' : processed_text_input, 'user_noun' : user_noun, 'orig_noun' : orig_noun, 'topics' : noun_topics, 'db_name' : db_name, 'generated' : using_generated}
        return ret_dict
    except KeyError:
        print("Error: Encountered unknown word.")
        return("Error getting reply")

def generated_reply_helper(input_sentence):
    
    output_sentence = agent.generate_reply(input_sentence)
    ans = agent.preprocess_reply(output_sentence[0])
    
    agent.generated_kb = agent.generated_kb.append({'question' : input_sentence, 'answer' : output_sentence[0], "processed_answer" : ans } , ignore_index=True)
    #agent.generated_kb.to_csv(r'data/generated_answers_kb.csv', index = False)
    #print(agent.generated_kb.answer)
    #print(ans, "GN")
    return ans
    