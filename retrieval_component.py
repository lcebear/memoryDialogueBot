import pandas as pd

import ast
import random

import agent_commons as cmn

#Exceptions for saying "I told you already" are 'Hello' 'bye' 'how are you?' and things regarding 'today'
#More exceptions may need to be added
exception_qid = [92, 94, 138, 139, 140, 84, 85, 78, 79, 82, 80 ]

#Read in template questions,answers and memory
retrieval_q = pd.read_csv('data/retrieval_questions.csv')
retrieval_a = pd.read_csv('data/retrieval_answers.csv')
user_history = pd.read_csv('data/user_history.csv')


#Convert 'string' to  list
for i in range(len(user_history)):
    user_history['message_history'].loc[i] = ast.literal_eval(user_history['message_history'].loc[i])
    
#fetching retrieval questions with their question id, once so we don't have to repeat this operation
retrieval_question_l = []
retrieval_qid_l = []
#only need to retrieve these once
for i in range(len(retrieval_q)):
    retrieval_question_l.append(retrieval_q.question[i])
    retrieval_qid_l.append(retrieval_q.answer_id[i])

global retrieval_embeddings
with cmn.g.as_default():
    global retrieval_embeddings
    retrieval_embeddings = cmn.sim_sess.run(cmn.embedded_text, feed_dict={cmn.text_input: retrieval_question_l})
    
    
#Turn the user's input into a vector and find the most similar question in KB
#return a random answer with the related answer_id to the found question.
def find_question_n_answer_retrieval(user_input):
    max_sim = 0
    max_sim_q = None
    answer_id = 0
    answer = None
    #give a list of embeddings and a new sentence. 
    #the max similarity is returned a long with the index to translate the vector embedding into a question.
    q_index, max_sim = cmn.universal_similarity(user_input, retrieval_embeddings)
    max_sim_q = retrieval_question_l[q_index]
    answer_id = retrieval_qid_l[q_index]
            
    fetch_answer = retrieval_a.loc[retrieval_a['answer_id'] == answer_id]
    second_answer = retrieval_a.loc[retrieval_a['optional_id'] == answer_id] 
    fetch_answer = fetch_answer.append(second_answer)
    
    if len(fetch_answer) > 0:
        answer = fetch_answer.sample().iloc[0].answer
        
    return answer, max_sim, answer_id, max_sim_q
    
#Checks the msg history for saved retrieved questions and answers
def check_msg_history(msg_hist, question_id):
    ans = None
    #Some default replies entries are "" 
    #to add some chance to just repeat the answer instead of always saying "I told you already" + repeat
    default_replies = [ "Maybe you forgot.", "", "I think I told you earlier.",
    "", "I told you before.", "I told you previously.",
    "I guess you forgot.", "I thought I told you already.", ""]
    
    found_ans = ""
    found_answer = False 
    for question, answer, qid in msg_hist:
        if qid == question_id:
            if qid in exception_qid:
                continue
            else:
                found_ans = answer
                found_answer = True
                break
            
    if found_answer:
        ans = default_replies[random.randint(0, len(default_replies )-1)] + " " + found_ans
        ans = ans.strip()
    
    return ans 
    