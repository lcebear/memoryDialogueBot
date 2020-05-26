import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 

import numpy as np
import pandas as pd

import tensorflow as tf
import gpt_2_simple as gpt2

import agent_commons as cmn

import threading 
#counter for queueing users into different generative models 
global gen_counter
gen_counter = 0

model_1_lock = threading.Lock()
model_2_lock = threading.Lock()
model_3_lock = threading.Lock()
model_4_lock = threading.Lock()
counter_lock = threading.Lock()

idf_weights_df = pd.read_csv('data/dataset_idf_weights.csv')
idf_words = idf_weights_df.word.tolist()
weight_mean = np.mean(idf_weights_df.idf_weights.tolist())


model_name = '124M'
#run_name ="run_18_pc_history"

run_names = ["run_18_pc_history", "run_18_pc_history_2", "run_18_pc_history_3", "run_18_pc_history_4"]
scopes = [ None, "m2", "m3", "m4"]
#Load the generative model 
#sess = gpt2.start_tf_sess(threads=8)

#global graph
#graph = tf.compat.v1.get_default_graph()

#with graph.as_default():
#    gpt2.load_gpt2(sess, run_name=run_name)

#Max 4 currently
num_models = 1
global graphs
gen_sessions = [None] * num_models
graphs = [None] * num_models
for i in range(num_models):
    
    #Load the generative model 
    gen_sessions[i] = gpt2.start_tf_sess(threads=8)
    #I don't remember why but reset is needed when loading >1 models 
    gen_sessions[i] = gpt2.reset_session(gen_sessions[i], threads=8)
    
    graphs[i] = tf.compat.v1.get_default_graph()
    with graphs[i].as_default():
        gpt2.load_gpt2(gen_sessions[i], run_name=run_names[i], scope=scopes[i])
    
     

#Update a word embedding (encoded sentence) to be old_vector*alpha + new_vector+(1-alpha)
def update_history_embedding(history_emb, new_entry, alpha=0.2):
    with cmn.g.as_default():
        result = cmn.sim_sess.run(cmn.embedded_text, feed_dict={cmn.text_input: [new_entry]})
        history_emb =  history_emb*alpha + result[0]*(1-alpha)
    return history_emb
    
    
 #A normalize function to get values in range of 0 to 1
def normalize_data(in_list):
    arr = np.array(in_list)
    denom= arr.max() - arr.min()
    if np.absolute(denom) < 0.00001:
        denom = 0.00001
    arr = (arr - arr.min())/denom
    return arr
    
    
def answer_idf_score(in_answers):
    scores = []
    for i in range(len(in_answers)):
        ans = in_answers[i]
        temp_score = 0
        tokens = word_tokenize(ans)
        words =[word.lower() for word in tokens if word.isalpha()]
        temp_lenght = 0
        for w in words:
            if w in idf_words:
                temp_lenght +=1
                temp_score +=idf_weights_df.loc[idf_weights_df.word == w].iloc[0].idf_weights
        if(temp_lenght >0):   
            temp_score /=temp_lenght
        scores.append(temp_score)
    scores/=weight_mean
    return scores

#Trying to avoid exact same answer twice in a row
def similar_answer_removal(new_answers, old_ans):
    #print(new_answers)
    threshold = float(2/3)
    temp_answers = []
    try: 
        old_tokens = nltk.word_tokenize(old_ans)
        old_words=[word.lower() for word in old_tokens if word.isalpha()]
        
        data =[]
        for doc in new_answers:   
            doc_tokens = nltk.word_tokenize(doc)
            words=[word.lower() for word in doc_tokens if word.isalpha()]
            data.append(words)

        d_c_list = []
        #for each answer...., find how many tokens are the same as previous answer
        for i in range(len(data)):
            d_c_list.append(0)
            temp_set = set()
            #for each token, check if token exists in previous answer
            for j in range(len(data[i])):
                if data[i][j] in old_words:
                    if data[i][j] in temp_set:
                        continue
                    else:
                        temp_set.add(data[i][j])
                        d_c_list[i] =d_c_list[i] +1
            #if the number of similar tokens/total tokens are above threshold, sentence is deemed too similar
            
            if len(data[i])>0 and float(d_c_list[i])/float(len(data[i])) > threshold:
                continue
            else:
                temp_answers.append(new_answers[i])
    except:
        print("Exception happend in similar_answer_removal")
        temp_answers=new_answers
    finally:
        return temp_answers


def question_answer_shared_word_penalty(answers, question):
    penalty_list = []
    stop_words = set(stopwords.words('english')) 
    
    q_tokens = nltk.word_tokenize(question)
    q_words=[word.lower() for word in q_tokens if word.isalpha()]
    temp = []
    for word in q_words:    
        if word not in stop_words:
            tokens = cmn.nlp(word) #
            for token in tokens:
                temp.append(token.lemma_)
    q_words = temp
    #print(q_words)
    data =[]
    for doc in answers:   
        doc_tokens = nltk.word_tokenize(doc)
        words=[word.lower() for word in doc_tokens if word.isalpha()]
        temp = []
        for word in words:    
            if word not in stop_words:
                tokens = cmn.nlp(word) #
                for token in tokens:
                    temp.append(token.lemma_)
        words = temp
        data.append(words)
    #print(data)
    d_c_list = []
    #for each answer...., find how many tokens are the same as previous answer
    q_len = len(q_words)
    for i in range(len(data)):
        d_c_list.append(0)
        temp_set = set()
        len_data = len(data[i])
        #for each token, check if token exists in previous answer
        for j in range(len(data[i])):
            if data[i][j] in q_words:
                if data[i][j] in temp_set:
                    continue
                else:
                    temp_set.add(data[i][j])
                    d_c_list[i] =d_c_list[i] +1
        len_common = d_c_list[i]
        if np.absolute(q_len*len_data) > 0.001:
            val = (len_common*len_common)/(q_len*len_data)
            const = 2
            val_ln = np.log(const + val) - np.log(const)
            penalty_list.append(val_ln)
        else:
            penalty_list.append(0)
        #print(data[i], d_c_list[i], val_ln)
    return penalty_list
        



#sim - penalty_list[i]*omega ,omega=0.3

#Simple ranking without penalty
def find_max_scored_answer(scores, answers):
    assert len(scores) == len(answers), "Lists are not equal lenght"
    
    max_score = -999
    ans = None
    for i in range(len(scores)):
        score_val = scores[i]
        if score_val > max_score:
            max_score = score_val
            ans = answers[i]
    return ans, max_score

#given a score (similarit score), a penalty is given to the score based on the answer's length
# % of penalty determined by "hyper"parameter beta
def answer_lenght_penalty(scores, answers, ideal_tokens=20, beta=0.8):

    norm_scores = normalize_data(scores)
    token_scores = []

    for i in range(len(answers)):
        tokens = word_tokenize(answers[i])
        n_tokens = len(tokens)
        
        log_val = np.absolute(ideal_tokens - n_tokens)
        if log_val < 1:
            log_val = 1
        token_scores.append(np.log(log_val))
    norm_log_token_score = normalize_data(token_scores)
    
    penalized_score = norm_scores*beta - norm_log_token_score*(1-beta)

    return penalized_score
 
 
#a penalty score is calculated based on how much the answer's lenght differs from the "ideal lenght".
# % of penalty determined by "hyper"parameter beta #, beta=0.8 #norm_scores*beta - norm_log_token_score*(1-beta)
def answer_lenght_penalty(answers, ideal_tokens=20):

    penalized_score = []

    for i in range(len(answers)):
        tokens = word_tokenize(answers[i])
        n_tokens = len(tokens)
        
        log_val = np.absolute(ideal_tokens - n_tokens)
        if log_val < 1:
            log_val = 1
        penalized_score.append(np.log(log_val))

    return penalized_score
    
    
#measure similarity between answers and a pre-existing embedding, returns list of similarity scores
def answer_emb_similarity_score(history_emb, answers):
    similarity_score = []
    
    with cmn.g.as_default():
        result = cmn.sim_sess.run(cmn.embedded_text, feed_dict={cmn.text_input: answers})
        
    for i in range(len(result)):
        #"Outputs from universal sentence encoder are roughly normalized vectors"
        sim = np.inner(result[i],history_emb)
        similarity_score.append(sim[0])
    return similarity_score

#TODO needs to be updated
def preprocess_reply(input_text):

    replaced = input_text.replace('<|startoftext|>', '')
    temp_split = replaced.split('endoftext', maxsplit=1)
    if len(temp_split) >1:
        replaced = temp_split[0]
        replaced = replaced.replace('<|', '')
    replaced = replaced.replace('<|endoftext|>', '')
    replaced = replaced.replace('?', '? ')
    text_sentences = cmn.nlp(replaced)
    temp_saved =""
    temp2 = ""
    has_answer = False
    for sentence in text_sentences.sents:
        if has_answer:
            temp_saved=temp2
            break
        if len(sentence)>1:
            temp2 = temp_saved
            temp_saved = temp_saved + " " + sentence.text
            for token in sentence:
                if token.text =="?":
                    if len(temp2) > 1:
                        has_answer = True
                        break
                    else:
                        temp_saved=""
                        temp2=""

    temp_saved = temp_saved.replace('<|', '')
    temp_saved = temp_saved.replace('|>', '')
    text_sentences = sent_tokenize(temp_saved)
    final =""
    count = 0
    
    #allow only up to two sentences in the reply.
    for sent in text_sentences:
        count +=1
        #If the second sentence doesn't end with a punctuation or exclamation
        #then perhaps the sentence was not fully generated and should be disregarded
        if count == 2:
            tokens = word_tokenize(sent)
            if len(tokens) > 0 and (tokens[-1] != '.' and tokens[-1] != '!'):
                break
        if count >2:
            break
        final = final + " " + sent
    return final.strip()


def generate_model(text_input, num_answers=8, index=0):
    gen_ans =[]
    try:
        with graphs[index].as_default():
            gen_ans =gpt2.generate(gen_sessions[index],
                          run_name=run_names[index],
                          length=40,
                          temperature=1,
                          prefix=text_input,
                          truncate="<|endoftext|>",
                          include_prefix=False,
                          nsamples=num_answers,
                          batch_size=num_answers,
                          top_p=0.9,
                          return_as_list=True,
                          scope=scopes[index]
                          )
    except Exception as e:
        print(e)
    
    return gen_ans

# generate_reply(qa_history_embedding, user_question, question_label,q_past_lab, ans_past_lab,question, previous_answer=None, num_answers=8):
def generate_reply(text_input, num_answers=8):
    temp_gen_counter = 0
    global gen_counter
    
    gen_ans =[]

    counter_lock.acquire()
    try:
        #update shared counter
        gen_counter = gen_counter + 1
        #store a local counter
        temp_gen_counter = gen_counter
        #resets counter 
        if gen_counter == num_models:
            gen_counter = 0
    finally:
        counter_lock.release()
    #TODO Check which model to acquire
    #Only one model now, so acquire the lock for it
    if temp_gen_counter == 1:
        model_1_lock.acquire()
        try:
            gen_ans= generate_model(text_input, num_answers=num_answers, index=0)
        finally:
            print("Released model 1")
            model_1_lock.release()
    elif temp_gen_counter == 2:
        model_2_lock.acquire()
        try:
            gen_ans= generate_model(text_input, num_answers=num_answers, index=1)
        finally:
            print("Released model 2")
            model_2_lock.release()
    elif temp_gen_counter == 3:
        model_3_lock.acquire()
        try:
            gen_ans= generate_model(text_input, num_answers=num_answers, index=2)
        finally:
            print("Released model 3")
            model_3_lock.release()
    elif temp_gen_counter == 4:
        model_4_lock.acquire()
        try:
            gen_ans= generate_model(text_input, num_answers=num_answers, index=3)
        finally:
            print("Released model 4")
            model_4_lock.release()
    
    return gen_ans 
    
    




#removes empty answers and answers who are too similar to previous answer, also preprocess answers.    
def bad_answer_removal(answers, previous_answer=None):
    temp_ans = []
    #Trying to prevent empty reply, preprocess replies
    for i in range(len(answers)):
        temp_string = answers[i].strip()
        if len(temp_string) < 2:
            continue
        else:
            pp_r =preprocess_reply(answers[i])
            if pp_r != None:
                temp_ans.append(pp_r)
    #EXCEPTION - if for some reason no answers remain after preprocessing    
    if len(temp_ans) == 0:
        temp_ans = answers
        print("WARNING: Generative model resulted in 0 valid replies")

    
    #remove similar or same answers
    remaining_ans =[]
    if previous_answer != None:
        remaining_ans = similar_answer_removal(temp_ans, previous_answer)
    if len(remaining_ans) > 0:
        temp_ans = remaining_ans
    
    return temp_ans 


#Rank the answer
def answer_ranking(history_emb, answers, question):
    #print('\n')
    beta = 0.8
    omega = 0.3
    theta = 0.2

    #"cosine similarity" between answers and word embedding 
    similarity_score = answer_emb_similarity_score(history_emb, answers)
    
    #for i in range(len(answers)):
    #    print(answers[i],similarity_score[i])
    
    #print('\n')
    
    #get penalty score based on word length  
    length_penalty_score = answer_lenght_penalty(answers, ideal_tokens=20)
    #get penalty score based on if answer reuse question keywords 
    keyword_penalty_score = question_answer_shared_word_penalty(answers, question)
    #reward rare word occurances
    idf_reward_score = answer_idf_score(answers)
    
    #print(length_penalty_score)
    
    #print(keyword_penalty_score)
    
    #print(idf_reward_score)
    
    
    temp_ans, temp_score = find_max_scored_answer(similarity_score, answers)
    #print("Answer after similarity:", temp_ans, temp_score)
    
    calc_score = normalize_data(similarity_score)*beta -normalize_data(length_penalty_score)*(1-beta)

    temp_ans, temp_score = find_max_scored_answer(calc_score, answers)
    #print("Answer after length penalty:", temp_ans, temp_score)
    
    #beta=0.8 #norm_scores*beta - norm_log_token_score*(1-beta)
    calc_score -= np.array(keyword_penalty_score)*omega
    temp_ans, temp_score = find_max_scored_answer(calc_score, answers)
    #print("Answer after keyword penalty:", temp_ans, temp_score)
    
    calc_score += normalize_data(idf_reward_score)*theta
    temp_ans, temp_score = find_max_scored_answer(calc_score, answers)
    #print("Answer after idf reward:", temp_ans, temp_score)
    
    
    final_answer, final_score = find_max_scored_answer(calc_score, answers)
    return final_answer
    
