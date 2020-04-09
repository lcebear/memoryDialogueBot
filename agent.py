import pandas as pd
import spacy
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import requests
import nltk
from pattern.en import pluralize, singularize
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import time
from timeit import default_timer as timer
import threading 

import gpt_2_simple as gpt2
import atexit

global gen_counter
gen_counter = 0
my_lock = threading.Lock()





# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()


# Create session and initialize.
sim_sess = tf.Session(graph=g)
sim_sess.run(init_op)


def exit_handler():
    print('My application is ending! Saving data')
    generated_kb.to_csv(r'data/generated_answers_kb.csv', index = False)
    like_memory = pd.read_csv('data/sentiment_memory.csv')

atexit.register(exit_handler)


nlp = spacy.load('en_core_web_sm')


#Read in template questions,answers and memory
template_q = pd.read_csv('data/likes_question_templates.csv')
retrieval_q = pd.read_csv('data/questions_templates.csv')
template_a = pd.read_csv('data/answer_templates.csv')
retrieval_a = pd.read_csv('data/answer_templates_2.csv')
like_memory = pd.read_csv('data/sentiment_memory.csv')
generated_kb = pd.read_csv('data/generated_answers_kb.csv')


#Some fruits are listed under "Food" topic, so the line below is temporary remove solution
like_memory = like_memory.drop_duplicates(subset='subject', keep="last")
#Assign random sentiment to every noun item in memory
temp = np.random.random(len(like_memory))
like_memory['sentiment'] = temp
like_memory['lc_subject'] = np.nan

#fetching retrieval questions with their question id, once so we don't have to repeat this operation
retrieval_question_l = []
retrieval_qid_l = []
#only need to retrieve these once
for i in range(len(retrieval_q)):
    #q = retrieval_q.question[i]
    #qid = retrieval_q.answer_id[i]
    retrieval_question_l.append(retrieval_q.question[i])
    retrieval_qid_l.append(retrieval_q.answer_id[i])
    
likes_question_l = []
likes_qid_l = []
#only need to retrieve these once
for i in range(len(template_q)):
    likes_question_l.append(template_q.question[i])
    likes_qid_l.append(template_q.answer_id[i])

global retrieval_embeddings, likes_embeddings   
with g.as_default():
    global retrieval_embeddings
    retrieval_embeddings = sim_sess.run(embedded_text, feed_dict={text_input: retrieval_question_l})
    likes_embeddings = sim_sess.run(embedded_text, feed_dict={text_input: likes_question_l})

#def calculate_topic_sent(): 
#Calculate topic average sentiment (not very useful considering random function mean 0.5)
topic_sent = {}
memory_topics = set(like_memory.topic)
for topic in memory_topics:
    topic_list = like_memory.loc[like_memory['topic'] == topic]
    count = 0
    divisor = len(topic_list)
    for i in range(divisor):
        count = count + topic_list.sentiment.iloc[i]
    topic_sent[topic] = count/divisor
#print(topic_sent)

#Extract a number of favorite noun's for each topic for easy access
topic_favorites = {}
select_n = 5

for topic in memory_topics:
    topic_list = like_memory.loc[like_memory['topic'] == topic]
    topic_list = topic_list.sort_values(by=['sentiment'], ascending=False)
    temp_l = []
    for i in range(select_n):
        temp_l.append(topic_list.subject.iloc[i])#,topic_list.sentiment.iloc[i]))
    topic_favorites[topic] = temp_l
  
#print(topic_favorites)
#memory_topics = topic_favorites.keys()


topic_dislike = {}
for topic in memory_topics:
    topic_list = like_memory.loc[like_memory['topic'] == topic]
    topic_list = topic_list.sort_values(by=['sentiment'])
    temp_l = []
    for i in range(select_n):
        temp_l.append(topic_list.subject.iloc[i])#,topic_list.sentiment.iloc[i]))
    topic_dislike[topic] = temp_l
  
#print(topic_dislike)

#
subj =""
for i in range(len(like_memory)):
    subj = like_memory.subject.iloc[i]
    like_memory.lc_subject.iloc[i] = subj.lower()
#print(like_memory.head()) 
#print(topic_dislike)

#Default values
sentiment_opt_pos = ["like", "likes", "love", "loves"]
sentiment_opt_neg = ["dislikes", "dislike", "hate", "hates"]
sentiment_opt = sentiment_opt_pos + sentiment_opt_neg
wildcards = {"noun": '<noun>', "sentiment":'<sentiment>', "topic" : "<topic>",
             "agent_sentiment" : '<sentiment_1>', "noun_1" : '<noun_1>', "noun_2" : '<noun_2>',
             "noun_3" : '<noun_3>'}
question_sentiment = "like" #default sentiment is to ask if you like something


model_name = '124M'
run_name ="run_10"

sess = gpt2.start_tf_sess(threads=8)

global graph
graph = tf.compat.v1.get_default_graph()

with graph.as_default():
    gpt2.load_gpt2(sess, run_name=run_name)



def fetch_subject_sentiment(key):
    key = key.lower()
    ans_sent = None

    temp_l = like_memory.loc[like_memory['lc_subject'] == key]

    if len(temp_l) > 0:
        ans_sent = temp_l.sentiment.iloc[0]
        #print(temp_l) #if subject listed under multiple 
    #print(key, ans_sent)
    return ans_sent
    
#Input subject to find topic: Apple -> Food/Fruit, Currently disabled due to performance.
def fetch_noun_relations(noun):
    temp_noun_set = set()
    return temp_noun_set
    try:
        
        query_noun = noun
        api_path = 'http://api.conceptnet.io/query?start=/c/en/' + query_noun + '&rel=/r/IsA'
        
        obj = requests.get(api_path).json()
        #print(obj['edges'][0])
        
        #outer for traverses the edges, inner for traverses the content in the 'end' tag
        for j in range(len(obj['edges'])):
            #print("Description:", obj['edges'][j]['surfaceText'])
            for i in obj['edges'][j]['end']:
                #if i == 'label':
                 #   print("Label:", obj['edges'][j]['end'][i]) 
                #elif i == 'sense_label':
                 #   print("Sense_label:", obj['edges'][j]['end'][i])
                temp_string = obj['edges'][j]['end'][i]
                text = nlp(temp_string)
                  
                for token in text:
                    tag = nltk.pos_tag([token.text])
                    if token.pos_ == "NOUN" or tag == "NN" or tag == 'NNS':
                        temp_noun_set.add(token.text)
    except Exception as e:
        print(e)
    finally:
        #print(temp_noun_set)
        return temp_noun_set
  
#input noun-> find noun's IsA relations e.g Apple is a fruit -> compare IsA relations with existing topics.
def check_noun_topic_exist_memory(noun):
    temp_noun_set = fetch_noun_relations(noun)
    
    union_topics = []
    for topic in memory_topics:
        for item in temp_noun_set:
            if item == topic:
                union_topics.append(item)
    return union_topics
    
#Check if noun is a known subject in memory
def is_noun_existing_subject(noun):
    temp_l = like_memory.loc[like_memory['lc_subject'] == noun.lower()]
    if len(temp_l) > 0:
        return True
    else:
        return False
        
#Check if noun is a known topic in memory
def is_noun_existing_topic(noun):
    return noun in memory_topics

#TODO can be optimized to make use of gpu computing by getting the embedding for all questions at same time.
def similarity_universal_sentence_decoder(user_in, in_emb):
    #s_t = timer()
    #global embedded_text
    #global text_input
    max_sim = 0
    max_index = 0
    #question.append(user_in)
    with g.as_default():
        result = sim_sess.run(embedded_text, feed_dict={text_input: [user_in]})
    for i in range(len(in_emb)):
            sim = np.inner(in_emb[i],result[0])
            if sim > max_sim:
                max_sim = sim
                max_index = i
    #end_t = timer()
    #print(end_t - s_t, "Took this long in similarity func")
    return max_index, max_sim#np.inner(result[0],result[1])
    
def similarity_calc(X,Y):
    X = X.lower() #input(q).lower() 
    Y = Y.lower() #input(form_input).lower() 

    # tokenization 
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 

    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 

    # remove stop words from string 
    X_set = {w for w in X_list}# if not w in sw}  
    Y_set = {w for w in Y_list} #if not w in sw} 

    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0

    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

#used to more quickly find max similarity between likes template questions and user input. 
#If high enough similarity, then more time is invested in process_user_input(). 
#This is done to avoid calling conceptnet for unseen nouns-
#- when the template question with highest similarity is not high enough to be used.
def simple_process_user_input(user_input):
    user_input = user_input.lower()
    extracted_nouns = []
    form_input = user_input
    form_input_2 = user_input
    global question_sentiment
    question_sentiment = "like"
    noun = None
    prev_token = None
    sentiment_exist = False
    
    text = nlp(user_input)
    
    for token in text:
        #print(token.text)
        if token.text in sentiment_opt:
            question_sentiment = token.text
            sentiment_exist = True
            continue
            
        tag = nltk.pos_tag([token.text])
        if token.pos_ == "NOUN" or tag[0][1] == "NN" or tag[0][1] == 'NNS':
            noun = token.text
            if prev_token != None:
                temp_str = prev_token + " " + token.text
                noun = temp_str
                #break

        prev_token = token.text if token.pos_ == "NOUN" else None



    
    if noun != None:       
        form_input_2 = form_input_2.replace(noun, wildcards["topic"])
        form_input = form_input.replace(noun, wildcards["noun"])
    if sentiment_exist:
        form_input = form_input.replace(question_sentiment, wildcards['sentiment'])
        form_input_2 = form_input_2.replace(question_sentiment, wildcards['sentiment'])
        


    max_index, max_sim = similarity_universal_sentence_decoder(form_input, likes_embeddings)
    max_index_2, max_sim_2 = similarity_universal_sentence_decoder(form_input_2, likes_embeddings)
    if max_sim_2 > max_sim:
        max_index = max_index_2
        max_sim = max_sim_2
        
    max_sim_q = likes_question_l[max_index]
    answer_id = likes_qid_l[max_index]
    print(max_sim, question_sentiment, form_input, form_input_2, max_sim_q )
    return max_sim   

#process the user input 
def process_user_input(user_input):
    user_input = user_input.lower()
    extracted_nouns = []
    form_input = user_input
    global question_sentiment
    question_sentiment = "like"
    global like_memory
    noun = None
    orig_noun = None
    prev_token = None
    sentiment_exist = False
    noun_topics = []
    text = nlp(user_input)
    
    for token in text:
        if token.text in sentiment_opt:
            question_sentiment = token.text
            sentiment_exist = True
            
        tag = nltk.pos_tag([token.text])
        if token.pos_ == "NOUN" or tag[0][1] == "NN" or tag[0][1] == 'NNS':
            if prev_token != None:
                temp_str = prev_token + " " + token.text
                #print("RAN THE IF", temp_str)
                extracted_nouns.insert(0, (temp_str, temp_str))
            
            extracted_nouns.append((token.text, token.text))
            extracted_nouns.append((token.lemma_, token.text))
            extracted_nouns.append((pluralize(token.text), token.text))
            #TODO: take into consideration singular/plural, video game vs video games
        prev_token = token.text if token.pos_ == "NOUN" else None
        #print(token.text, token.pos_)
    for n in extracted_nouns:
        orig_noun = n[1]
        
        if is_noun_existing_topic(n[0]):
            noun_topics = [n[0]]
            noun = n[0]
            break
            
        if is_noun_existing_subject(n[0]):
            noun = n[0]
            noun_topics = [like_memory.loc[like_memory['lc_subject'] == noun].topic.iloc[0]]
            break
    #If the noun is not a recognized topic or subject...
    # TODO: Currently disabled. Perform this operation after generating an answer. 
    if noun == None:
        for n in extracted_nouns:
            #calls conceptNet to find the noun's IsA relations and checks if the relations == existing topic
            noun_topics = check_noun_topic_exist_memory(n[0])
            #if the topic exist but the noun is not known, add it to our list with random sentiment
            if len(noun_topics) >0:
                noun = n[0]
                orig_noun = n[1]
                #if the noun is not a known subject (Apple, Soccer, Pasta) then add it with random sentiment  
                noun_sent = np.random.random(1)
                for topic in noun_topics:
                    like_memory = like_memory.append(
                        {'subject' : noun , 'topic' : topic, 'sentiment' : noun_sent, 'lc_subject' : noun.lower()} ,
                        ignore_index=True)
                break
    
    if noun != None:
        if is_noun_existing_topic(noun):
            noun_topics = [noun]
            form_input = form_input.replace(orig_noun, wildcards["topic"])
        else:
            form_input = form_input.replace(orig_noun, wildcards["noun"])
    if sentiment_exist:
        form_input = form_input.replace(question_sentiment, wildcards['sentiment'])

    #print(user_input, form_input, noun, orig_noun, noun_topics)
    return form_input, noun, orig_noun, noun_topics 
    
def find_question_n_answer_retrieval(user_input):
    max_sim = 0
    max_sim_q = None
    answer_id = 0
    answer = None
    
        #cosine = similarity_universal_sentence_decoder(user_input, q)#similarity_calc(q,user_input)
    q_index, max_sim = similarity_universal_sentence_decoder(user_input, retrieval_embeddings)#, retrieval_question_l)
    max_sim_q = retrieval_question_l[q_index]
    answer_id = retrieval_qid_l[q_index]
        #if cosine > 0.98:
        #    max_sim_q = q
        #    answer_id = qid
        #    max_sim = cosine
        #    break
        #elif cosine > max_sim:
        #    max_sim = cosine
        #    max_sim_q = q
        #    answer_id = qid
            
    fetch_answer = retrieval_a.loc[retrieval_a['answer_id'] == answer_id]
    second_answer = retrieval_a.loc[retrieval_a['optional_id'] == answer_id] 
    fetch_answer = fetch_answer.append(second_answer)
    
    if len(fetch_answer) > 0:
        answer = fetch_answer.sample().iloc[0].answer
    #print(max_sim_q, max_sim)
    return answer, max_sim, answer_id, max_sim_q
    
#find a suitable question template and return it
def find_question_template(processed_text_input):
    #s_t = timer()
    max_sim = 0
    max_sim_q = None
    answer_id = 0
        
    max_index, max_sim = similarity_universal_sentence_decoder(processed_text_input, likes_embeddings)
    max_sim_q = likes_question_l[max_index]
    answer_id = likes_qid_l[max_index]
    #end_t = timer()
    #print("Took this long in find_question_template:", end_t - s_t)
    #print(max_sim_q, max_sim)
    return answer_id, max_sim, max_sim_q 

def fetch_answer_template(answer_id, noun, noun_topics):
    global like_memory
    global question_sentiment
    fetch_answer = template_a.loc[template_a['answer_id'] == answer_id]
    #default
    ans_sentiment = "like"
    ans_sent_val = None
    ret_nouns = noun 
    for key in memory_topics:
        if noun == key:
            if question_sentiment in sentiment_opt_pos:
                ret_nouns = topic_favorites[key]
                ans_sent_val = 0.5
                break
            else:
                ret_nouns = topic_dislike[key]
                ans_sent_val = 0.5
                #question_sentiment = "dislike"
                if answer_id != 1:
                    ans_sentiment = "hate"
                break
    #if the noun is not a topic (Food/Sports/...)
    if ans_sent_val == None:
        ans_sent_val = fetch_subject_sentiment(noun)
        ans_sentiment = sent_float_to_text(ans_sent_val)

    #the user is talking about something we don't handle in memory.
    elif ans_sent_val == None and noun_topics == None:
        #todo
        pass

        
            
    if (((ans_sentiment in sentiment_opt_pos) and (question_sentiment in sentiment_opt_pos)) 
        or ((ans_sentiment in sentiment_opt_neg) and (question_sentiment in sentiment_opt_neg))):
        fetch_answer = fetch_answer.loc[fetch_answer['same_sentiment'] == 1]
    else:
        fetch_answer = fetch_answer.loc[fetch_answer['same_sentiment'] == 0]
        
    #fetch_answer = template_a.loc[template_a['answer_id'] == answer_id ]
    return fetch_answer, ret_nouns, ans_sentiment
    
def sent_float_to_text(sentiment):
    ret_sentiment = "love"
    if sentiment < 0.1:
        ret_sentiment = "hate"
    elif sentiment < 0.5:
        ret_sentiment = "dislike"
    elif sentiment < 0.9:
        ret_sentiment = "like"
        
    return ret_sentiment
    
def process_agent_output(answer_template, noun, nouns, noun_topics, answer_sentiment):
    agent_output = answer_template.answer
    temp_nouns = nouns
    #print(agent_output, nouns, noun_topics, (nouns))
    if answer_template.fetch_count > 0 and noun_topics != None and len(noun_topics) >0:
        #print(noun_topics)
        if question_sentiment in sentiment_opt_pos:
            temp_nouns = topic_favorites[noun_topics[0]]
        elif question_sentiment in sentiment_opt_neg:
            temp_nouns = topic_dislike[noun_topics[0]]
        sing_noun = singularize(noun)
        plural_noun = pluralize(noun)
        if sing_noun in temp_nouns: temp_nouns.remove(sing_noun)
        elif plural_noun in temp_nouns: temp_nouns.remove(plural_noun)
            
    #replace nouns
    for i in range(1,answer_template.fetch_count+1):
        temp = "noun_"+str(i)
        
        agent_output = agent_output.replace(wildcards[temp], temp_nouns[i-1])
    
    if answer_template.use_noun:
        agent_output = agent_output.replace(wildcards["noun"], noun)
    if answer_template.use_sentiment:
        agent_output = agent_output.replace(wildcards["sentiment"], question_sentiment)
    agent_output = agent_output.replace(wildcards["agent_sentiment"], answer_sentiment)
    #print(agent_output)
    return agent_output
    
def generate_reply(user_question, num_answers=20):
    ret_num = 0
    #adding question mark if sentence doesn't have it
    temp = ""
    tokens = word_tokenize(user_question)
    
    try:
        if nltk.tag.pos_tag([tokens[-1]])[0][1] != '.':
            user_question = user_question + "?"
    except Exception as e:
        print("failed to add question mark to user sentence", e)
    
    global gen_counter
    print("In generate_reply", user_question)
    text_input = "<|startoftext|>" + user_question
    #Before prediction
    K.clear_session()
    my_lock.acquire()
    #try:
    try:
        #print("acq lock", gen_counter)
        gen_counter = gen_counter + 1
    finally:
        
        time.sleep(1*gen_counter)
        #print("release lock")
        my_lock.release()
    while(True):
        try: 
            with graph.as_default():
                gen_ans =gpt2.generate(sess,
                              run_name=run_name,
                              length=40,
                              temperature=1,
                              prefix=text_input,
                              truncate="<|endoftext|>",
                              include_prefix=False,
                              nsamples=num_answers,
                              batch_size=num_answers,
                              top_p=0.9,
                              return_as_list=True
                              )
            ret_num = 0
            #Trying to prevent empty reply
            for i in range(num_answers-1):
                temp_string = gen_ans[ret_num].strip()
                if len(temp_string) < 2:
                    ret_num+=1
                else:
                    break
            print(gen_ans[ret_num])
            break
        except Exception as e:
            print("Tensorflow thread error: Called gen in parallel", e, ret_num, gen_ans)#, threading.get_ident(), threading.enumerate())
            
            time.sleep(0.5)
    #finally:       
    K.clear_session()
    #  my_lock.release()
    my_lock.acquire()
    try:
        #print("acq lock")
        gen_counter = gen_counter - 1
    finally:
        #print("release lock")
        my_lock.release()
    return gen_ans[ret_num]

def preprocess_reply(input_text):

    replaced = input_text.replace('<|startoftext|>', '')
    temp_split = replaced.split('endoftext', maxsplit=1)
    if len(temp_split) >1:
        replaced = temp_split[0]
        replaced = replaced.replace('<|', '')
    replaced = replaced.replace('<|endoftext|>', '')
    replaced = replaced.replace('?', '?\n')
    text_sentences = nlp(replaced)
    temp_saved =""
    temp2 = ""
    has_answer = False
    for sentence in text_sentences.sents:
        if has_answer:
            temp_saved=temp2
            break
        #print(sentence.text, len(sentence))
        if len(sentence)>1:
            temp2 = temp_saved
            temp_saved = temp_saved + " " + sentence.text
            for token in sentence:
                #print(token.text, token.pos_)
                if token.text =="?":
                    if len(temp2) > 1:
                        has_answer = True
                        break
                    else:
                        temp_saved=""
                        temp2=""
            
    text_sentences = sent_tokenize(temp_saved)
    final =""
    count = 0
    for sent in text_sentences:
        count +=1
        if count >2:
            break
        final = final + " " + sent
    return final.strip()
    