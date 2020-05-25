import likes_component as likes

print("Likes component loaded")
import retrieval_component as retrieval

print("retrieval component loaded")

import generative_component as gen 

import numpy as np 
import pandas as pd
import time
from timeit import default_timer as timer

import nltk
from nltk.tokenize import word_tokenize

print("generative component loaded")

class_history = pd.read_csv('data/classified_user_history.csv')

#Threshold for similarity, if user input is below the thresholds, then generate an answer.
retrieval_threshold = 0.9 #Below this threshhold, the question is generated isntead of retrieved.
likes_threshold = 0.9

#2 == 1 question and answer pair
num_hist_msg = 2

max_hist_msg = 4
follow_up_q_labels = [3,9,10]
history_less_labels = [11, 18, 20, 22, 24, 27, 29, 30, 33, 34, 36, 70, 100]

def get_reply(input_sentence, user_id):
    try:
        global class_history
        start = timer()
        
        #---------
        #(TODO), store msg_hist and hist embedding in a csv with user_id as key
        #Not vital, as class_history keeps history and hist embedding can be remade from past msgs.
        qa_history_embedding = None
        msg_hist = []

        past_answer = None
        past_qlabel = 0

        qlabel = 0
        alabel = 0

        #---------
        user_msg_history = []
        use_past_msg_pair = True
        
        #return related 
        error_msg = None 
        answer = None
        
        #likes/retrieval related
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
            
            


        
        
        ## classify question
        qlabel = gen.cmn.classify_question(input_sentence)
        desc = gen.cmn.question_labels[qlabel]

       
        
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
        print(sim_val_2, max_sim_val) 
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
        
        if using_generated:
            #---------------
            class_user_hist = class_history.loc[class_history['userID'] == user_id]
            len_user_hist = len(class_user_hist)
            if len_user_hist >0:
                past_qlabel = class_user_hist.iloc[-1].qlabel
                past_answer = class_user_hist.iloc[-1].answer
                

                
            if qlabel in follow_up_q_labels:
                qlabel = past_qlabel
                
            if qlabel == past_qlabel:
                use_past_msg_pair = False
            # check if question class exist in history -> Pandas dataframe with question+answer+q_class,a_class
            topic_user_hist = class_user_hist.loc[class_user_hist['qlabel'] == qlabel]
            len_topic_hist = len(topic_user_hist)

            #input to history
            hist = "<|startofhistory|>"
            if len_topic_hist > 0 and qlabel not in history_less_labels:
                temp_count = 0
                if len_topic_hist > max_hist_msg:
                    temp_index = len_topic_hist - max_hist_msg
                else:
                    temp_index = 0
                for i in range(len_topic_hist):
                    if temp_count >= max_hist_msg:
                        break
                    temp_df = topic_user_hist.iloc[temp_index + i]
                    hist += "<|startoftext|>" + temp_df.question + " " + temp_df.answer + "<|endoftext|>\n"
                    #recalculate history embedding -> question -> answeer
                    qa_history_embedding = gen.update_history_embedding(qa_history_embedding, temp_df.question, alpha=0.2)
                    qa_history_embedding = gen.update_history_embedding(qa_history_embedding, temp_df.answer, alpha=0.2)
                    temp_count +=1
            
            
            if len_user_hist > 0 and use_past_msg_pair:
                past_msg_df = class_user_hist.iloc[-1]
                hist += "<|startoftext|>"+past_msg_df.question + " " + past_msg_df.answer + "<|endoftext|>\n"
                qa_history_embedding = gen.update_history_embedding(qa_history_embedding, past_msg_df.question, alpha=0.2)
                qa_history_embedding = gen.update_history_embedding(qa_history_embedding, past_msg_df.answer, alpha=0.2)
                

            print("Past answer:", past_answer)
            hist += "<|endofhistory|>"
                
            gen_input = hist + question_string

                
        #add question to history 
        msg_hist.append(question_string)

        #update history embedding -> Past answer embedding*0.2 + new question embedding*0.8
        #if use_past_msg_pair:
        qa_history_embedding = gen.update_history_embedding(qa_history_embedding, input_sentence, alpha=0.2)
        #else: #if it's a follow up question then use more weight of past messages than current question 
        #    qa_history_embedding = gen.update_history_embedding(qa_history_embedding, input_sentence, alpha=0.4)
        
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
        alabel = gen.cmn.classify_answers([answer])[0]
        past_answer = answer
        past_qlabel = qlabel
        answer_string = " " + answer + '<|endoftext|>'
        msg_hist.append(answer_string)
         
        qa_history_embedding = gen.update_history_embedding(qa_history_embedding, answer, alpha=0.2)
        class_history = class_history.append({"question" : input_sentence, "answer" : answer,
                                          "qlabel" : qlabel, "alabel" : alabel,
                                          "desc" : desc ,"userID" : user_id} , ignore_index=True)
        
        
        end = timer()
        print(end - start)
        
    except Exception as e:
        print(e)
        error_msg = e
    finally:
        return {"Error" : error_msg, "answer" : answer}
    #except Exception as e:
        #print(e, "Error: Encountered unknown word.")

    

