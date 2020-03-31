import agent

input_sentence = ''
similarity_threshhold = 0.8 #Below this threshhold, the question is generated isntead of retrieved.
def get_reply(input_sentence):
    try:

        processed_text_input, user_noun, orig_noun, noun_topics  = agent.process_user_input(input_sentence)
        answer_id, max_sim_val = agent.find_question_template(processed_text_input)[0:2]
        
        #Check the retrieval structure if a higher question similarity is found.
        answer, sim_val_2 = agent.find_question_n_answer_retrieval(input_sentence)[0:2]
        print(sim_val_2, max_sim_val)
        print(input_sentence)
        if (sim_val_2 < similarity_threshhold) and (max_sim_val < similarity_threshhold):
            ans = agent.generate_reply(input_sentence)
            print(ans[0], "GN")
            answer = ans[0]
        else:
            if sim_val_2 > max_sim_val:
                if answer != None:
                    print(answer)
                else:
                    ans = agent.generate_reply(input_sentence)
                    print(ans[0], "GN")
                    answer = ans[0]
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
                ans = agent.generate_reply(input_sentence)
                #print('\n',input_sentence)#, question_sentiment)
                print(ans[0], "GN")
                answer = ans[0]
        print(answer)
        return answer
    except KeyError:
        print("Error: Encountered unknown word.")
        return("Error getting reply")