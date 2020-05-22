import likes_component as likes

print("Likes component loaded")
import retrieval_component as retrieval

print("retrieval component loaded")

import generative_component as gen 

import numpy as np 
import time
from timeit import default_timer as timer

import nltk
from nltk.tokenize import word_tokenize

print("generative component loaded")


#Threshold for similarity, if user input is below the thresholds, then generate an answer.
retrieval_threshold = 0.85 #Below this threshhold, the question is generated isntead of retrieved.
likes_threshold = 0.9


input_sentence = ''
qa_history_embedding = None
msg_hist = []
hist = ""
#2 == 1 question and answer pair
num_hist_msg = 2

question_string = ""

past_answer = None

user_id = 1337
user_msg_history = []

using_generated = False
likes_has_answer = False

while(1):
    answer = None
    answer_id = 0
    max_sim_val = 0
    max_sim_q = None
    user_noun = None
    orig_noun = None
    noun_topics = None
    processed_text_input = None
    
    using_generated = False
    likes_has_answer = False
    #fetch the current user's entry from the dataframe if exist or add new entry
    curr_user = retrieval.user_history.loc[retrieval.user_history['userID'] == user_id]
    if len(curr_user) > 0:
        if len(curr_user.message_history.iloc[0]) > 0:
            user_msg_history = curr_user.message_history.iloc[0]        
    else:
        retrieval.user_history = retrieval.user_history.append({'userID' : user_id , 'message_history' : [], 'true_sentiment' : [] } , ignore_index=True)
        print("Added ", user_id, "to user history")
        curr_user = retrieval.user_history.loc[retrieval.user_history['userID'] == user_id]
        
        
    #try:
    # Get input sentence
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit': break
    start = timer()
   
    
    #----------------------------------------------------------------
    #Getting answer from likes component
    processed_text_input, user_noun, orig_noun, noun_topics  = likes.process_user_input(input_sentence)
    if user_noun != None:

        answer_id, max_sim_val = likes.find_question_template(processed_text_input)[0:2]
        
        answer, nouns, answer_sentiment = likes.fetch_answer_template(answer_id, user_noun, noun_topics)
        if len(answer) >0:
            answer = answer.sample().iloc[0]
            likes_has_answer = True
        else:
            answer = None 
    #----------------------------------------------------------------        
    #Getting answer from retrieval component 
    answer2, sim_val_2, answer_id_2, max_sim_q_2 = retrieval.find_question_n_answer_retrieval(input_sentence)
    #if both likes and retrieval component are below threshold in similarity score, use generative model
    if (sim_val_2 < retrieval_threshold) and (max_sim_val < likes_threshold):
        using_generated = True
    else:
        if sim_val_2 > max_sim_val: #if retrieval has higher similarity than likes component
                if answer2 != None:
                    #print(answer)
                    answer = answer2
                    max_sim_val = sim_val_2
                    answer_id = answer_id_2
                    previous_answer = retrieval.check_msg_history(user_msg_history, answer_id)
                    if previous_answer != None:
                        answer = previous_answer
                    #update user_msg_history with new retrieved q and a
                    else:
                        if answer_id in retrieval.exception_qid:
                            pass
                        else:
                            user_msg_history.append((input_sentence, answer2, answer_id))
                            #Todo, implement true_sentiment
                            retrieval.user_history.at[curr_user.index.values[0], 'message_history'] = user_msg_history
                #Retrieval similarity is higher than likes similarity but no answer was found.
                else:
                    using_generated = True
        #likes component has higher similarity than retrieval component 
        elif likes_has_answer:
            answer = likes.process_agent_output(answer,
                                         orig_noun, nouns,noun_topics, answer_sentiment)
        #The likes/dislikes template cannot handle the user input, introduce generative model.    
        else:
            using_generated = True

        
    #---------------------------------------------------------
        
    #Runs once, for the first sentence.
    if type(qa_history_embedding) is not np.ndarray:
        with gen.cmn.g.as_default():
            qa_history_embedding = gen.cmn.sim_sess.run(gen.cmn.embedded_text, feed_dict={gen.cmn.text_input: [input_sentence]})

    #process input to generative model
    question_string ="<|startoftext|>"+input_sentence

    #adding question mark if sentence doesn't have it
    tokens = word_tokenize(input_sentence)
    if nltk.tag.pos_tag([tokens[-1]])[0][1] != '.':
        question_string = question_string + "?"
    
    hist = "<|startofhistory|>" + "<|endofhistory|>"
    if len(msg_hist)>= num_hist_msg:
        hist = "<|startofhistory|>"
        for y in range(num_hist_msg):
            x = num_hist_msg -y
            hist += msg_hist[-x] +'\n'
        hist += "<|endofhistory|>"
        
    gen_input = hist + question_string

            
    #add question to history 
    msg_hist.append(question_string)

    #update history embedding -> Past answer embedding*0.2 + new question embedding*0.8
    qa_history_embedding = gen.update_history_embedding(qa_history_embedding, input_sentence, alpha=0.2)
    
    if using_generated: 
    
        print(gen_input)
        #Generate answers
        generated_answers = gen.generate_reply(gen_input)
        #remove bad answers (empty or too similar to past answer)
        generated_answers = gen.bad_answer_removal(generated_answers, past_answer)
        #rank answers
        output_sentence = gen.answer_ranking(qa_history_embedding, generated_answers, input_sentence)
            
        print(output_sentence, '\n')

        answer = output_sentence

    else:
        print(answer, '\n')
        
    #update past answer, append to history  

    past_answer = answer
    answer_string = " " + answer + '<|endoftext|>'
    msg_hist.append(answer_string)
     
    qa_history_embedding = gen.update_history_embedding(qa_history_embedding, answer, alpha=0.2)
    end = timer()
    print(end - start)
        
    #except Exception as e:
        #print(e, "Error: Encountered unknown word.")

    

