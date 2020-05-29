import pandas as pd
import numpy as np
import tensorflow as tf
import requests

#NLP
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
#from nltk.tokenize import sent_tokenize
from pattern.en import pluralize, singularize

#print("Importing commons")
import agent_commons as cmn
#print("Done importing commons")


#----------------------------------------------------------------

#load csv files 
template_q = pd.read_csv('data/question_templates.csv')
template_a = pd.read_csv('data/answer_templates.csv')
like_memory = pd.read_csv('data/sentiment_memory.csv')
disclosure_df = pd.read_csv('data/disclosure_answer_templates.csv')
disclose_reflect_df = pd.read_csv('data/disclosure_and_reflect_templates.csv')

#You can randomize sentiment every time or have user specific sentiment
#for last user test i'm commenting out the random sentiment
#incase the agent needs to be restarted
#temp = np.random.random(len(like_memory))
#like_memory['sentiment'] = temp #randomize sentiment (randomize preference/persona)

like_memory['sentiment'] = pd.to_numeric(like_memory['sentiment'])
like_memory['lc_subject'] = np.nan #lowercase subject

likes_question_l = []
likes_qid_l = []
#only need to retrieve these once, retrieves all questions and their qid and puts them into list
for i in range(len(template_q)):
    likes_question_l.append(template_q.question[i])
    likes_qid_l.append(template_q.answer_id[i])
    
#Store the topics that the agent handles
memory_topics = set(like_memory.topic)
memory_subjects = set(like_memory.subject)

#Turn the template questions into word embeddings
global likes_embeddings  
with cmn.g.as_default():
    global likes_embeddings
    likes_embeddings = cmn.sim_sess.run(cmn.embedded_text, feed_dict={cmn.text_input: likes_question_l})

#Get the spaCy tokens for the topics.
topic_tokens = '. '.join(map(str, memory_topics)) 
topic_tokens = cmn.nlp(topic_tokens)
#Get the spaCy tokens for the subjects.
subject_tokens = '. '.join(map(str, memory_subjects)) 
subject_tokens = cmn.nlp(subject_tokens)


#Extract a number of favorite noun's for each topic for easy access
def extract_nouns_by_sentiment(inp_dict, select_n=5, asc_order=False):
    temp_dict = {}
    for topic in memory_topics:
        topic_list = like_memory.loc[like_memory['topic'] == topic]
        topic_list = topic_list.sort_values(by=['sentiment'], ascending=asc_order)
        temp_l = []
        for i in range(select_n):
            temp_l.append(topic_list.subject.iloc[i])
        temp_dict[topic] = temp_l
    return temp_dict
 
topic_favorites = {}
topic_dislike = {}
topic_favorites = extract_nouns_by_sentiment(topic_favorites)
topic_dislike = extract_nouns_by_sentiment(topic_dislike, asc_order=True)

#adding lowercase subject (To be able to compare user input to agent memory)
subj =""
for i in range(len(like_memory)):
    subj = like_memory.subject.iloc[i]
    like_memory.lc_subject.iloc[i] = subj.lower()
 
 
#Default values
sentiment_opt_pos = ["like", "likes", "love", "loves"]
sentiment_opt_neg = ["dislikes", "dislike", "hate", "hates"]
sentiment_opt = sentiment_opt_pos + sentiment_opt_neg
wildcards = {"noun": '<noun>', "sentiment":'<sentiment>', "topic" : "<topic>",
             "agent_sentiment" : '<sentiment_1>', "noun_1" : '<noun_1>', "noun_2" : '<noun_2>',
             "noun_3" : '<noun_3>', "sing_noun" : '<sing_noun>', "sing_topic" : '<sing_topic>',
             "plural_noun" : '<plural_noun>', "plural_topic" : '<plural_topic>'}
question_sentiment = None
user_input_sentiment = None


#Given user input, find the most similar entry in agent memory
def find_similar_noun(user_in, agent_mem_tokens):
    max_sim = 0
    max_match = None
    #print("User IN", user_in)
    #print("Came to find similar noun", user_in )
    if user_in == None:
        return max_match, max_sim
    user_token = cmn.nlp(user_in)
    if (user_token and user_token.vector_norm):
        for token in agent_mem_tokens.sents:
            if (token and token.vector_norm):
                sim = user_token.similarity(token)
                if sim > max_sim:
                    max_match = token.text
                    max_sim = sim
                #print("Printing sim:", token.text, token, max_match)
    #print("User EXIT")
    if max_match != None:
        max_match = max_match.replace(".", "")
    return max_match, max_sim


#Find key in agent's subject memory, retrieve the sentiment value for it
#Function makes use of -> find_similar_noun(), like_memory
def fetch_subject_sentiment(key):
    key = key.lower()
    ans_sent = None
    threshold = 0.85
    temp_l = like_memory.loc[like_memory['lc_subject'] == key]

    if len(temp_l) > 0:
        ans_sent = temp_l.sentiment.iloc[0]
        #print("IN if")
    #If the key was not found, find the most similar entry, if it's above threshold, return its sentiment
    else:
        
        temp_match, match_sim = find_similar_noun(key, subject_tokens)
        #print(temp_match)
        #if match_sim > threshold:
        temp_l = like_memory.loc[like_memory['subject'] == temp_match]
        if len(temp_l) > 0:
            ans_sent = temp_l.sentiment.iloc[0]
        

    return ans_sent
    
    
#Input subject to find topic: Apple -> Food/Fruit, using word embedding similarity
#Function makes use of -> find_similar_noun()
def fetch_noun_relations(noun):
    temp_noun_set = set()
    
    #calculate word embedding similarity
    topic, max_sim = find_similar_noun(noun, topic_tokens)
    temp_noun_set.add(topic)
    return temp_noun_set
    
    
#Input subject to find topic: Apple -> Food/Fruit, using conceptnet
def concept_fetch_noun_relations(noun):
    temp_noun_set = set()
    try:
        query_noun = noun.strip()
        api_path = 'http://api.conceptnet.io/query?start=/c/en/' + query_noun + '&rel=/r/IsA'
        obj = requests.get(api_path).json()

        max_traversal = 3
        #outer for traverses the edges, inner for traverses the content in the 'end' tag
        for j in range(min(3,len(obj['edges']))):
            #print("Description:", obj['edges'][j]['surfaceText'])
            #From end tag, get text in labels and extracts nouns, adds nouns to set
            for i in obj['edges'][j]['end']:
                #if i == 'label':
                 #   print("Label:", obj['edges'][j]['end'][i]) 
                #elif i == 'sense_label':
                 #   print("Sense_label:", obj['edges'][j]['end'][i])
                temp_string = obj['edges'][j]['end'][i]
                text = word_tokenize(temp_string)
                for token in text:
                    tag = nltk.pos_tag([token])
                    if tag[0][-1] == "NN" or tag[0][-1] == 'NNS':
                        temp_noun_set.add(token)

    except Exception as e:
        print(e)
    finally:        
        return temp_noun_set
        
        
#Todo function that gets description "What is x?", This is more of a task-oriented issue
def concept_get_noun_description(noun):
    desc =None
    try:
        query_noun = noun.strip()
        api_path = 'http://api.conceptnet.io/query?start=/c/en/' + query_noun + '&rel=/r/IsA'
        obj = requests.get(api_path).json()

        max_traversal = 3
        #outer for traverses the edges, inner for traverses the content in the 'end' tag
        for j in range(min(3,len(obj['edges']))):
            if obj['edges'][j]['surfaceText']:
                desc = obj['edges'][j]['surfaceText']
    except Exception as e:
        print(e)

    finally:
        return desc


#input noun-> find noun's IsA relations e.g Apple is a fruit -> compare IsA relations with existing topics.
#Function makes use of -> fetch_noun_relations(), (can make use of conceptnet instead but slower)
def check_noun_topic_exist_memory(noun):
    #Find similar (or the) topic
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
        
    
#process the user input 
def process_user_input(user_input):
    #make it lower case, 
    user_input = user_input.lower()
    #TODO-> Expand contractions
    #Get sentiment value of user input
    global user_input_sentiment, question_sentiment, like_memory
    user_input_sentiment = cmn.user_sentiment(user_input)
    #Text sentiment ("Like"/"Dislike")
    question_sentiment = None
    
    #Stores all the nouns found in the sentence
    extracted_nouns = []
    form_input = user_input
    noun = None
    orig_noun = None
    prev_token = None
    sentiment_exist = False
    noun_topics = []
    max_match = None
    
    text = cmn.nlp(user_input)
    #Look through each token, look for nouns in the text
    for token in text:
        if token.text in sentiment_opt:
            question_sentiment = token.text
            sentiment_exist = True
            
    extracted_nouns = cmn.extract_user_nouns(text)
     
    #go through the list of extracted nouns
    for n in extracted_nouns:
        orig_noun = n[1]
        #first check if the noun is an existing topic, if it is then assume a question of type
        #What (food) do you like?
        
        #Updated: Changed order, used to check for topic first...unsure if order affects anything.
        #Reason: added "if max_sim > 0.65 -> assume topic"
        if is_noun_existing_subject(n[0]): #checks if noun is in list of subjects
            noun = n[0]
            noun_topics = [like_memory.loc[like_memory['lc_subject'] == noun].topic.iloc[0]]
            break
        
        if is_noun_existing_topic(n[0]): #checks if noun is in list of topics
            noun_topics = [n[0]]
            noun = n[0]
            break
        else:
            topic, max_sim = find_similar_noun(n[0], topic_tokens)
            #if similarity is higher than some threshhold, the noun is considered a topic
            if max_sim > 0.65: #e.g tv series, show, colour -> similarity to existing topic above 0.65
                noun_topics = [topic]
                noun = topic
                break

    #If the noun is not a recognized topic or subject...
    if noun == None:
        #df_lock.acquire()
        try:
            #max_match = None
            max_sim = 0
            for n in extracted_nouns:
                #print("looping extracted_nouns", n)
                match, sim = find_similar_noun(n[0], subject_tokens)
                if sim > max_sim:
                    max_match = match
                    noun = n[0]
                    orig_noun = n[1]
                    #print("In processing:",max_match, noun, orig_noun)

            #TODO: (Potentially) May only want to consider as topic if similarity is above threshold
            if noun != None:
                noun_topics = check_noun_topic_exist_memory(noun)
            #if the topic exist but the noun is not known, add it to our list with random sentiment
                if len(noun_topics) >0:
                    #if the noun is not a known subject (Apple, Soccer, Pasta) then add it with random sentiment  
                    noun_sent = np.random.random(1)
                    for topic in noun_topics:
                        like_memory = like_memory.append(
                            {'subject' : noun , 'topic' : topic, 'sentiment' : noun_sent[0], 'lc_subject' : noun.lower()} ,
                            ignore_index=True)
        finally:
            #df_lock.release()
            pass
    if noun != None:
        if is_noun_existing_topic(noun):
            noun_topics = [noun]
            form_input = form_input.replace(orig_noun, wildcards["topic"])
        else:
            form_input = form_input.replace(orig_noun, wildcards["noun"])
    if sentiment_exist:
        form_input = form_input.replace(question_sentiment, wildcards['sentiment'])
        
    question_sentiment = "like" if user_input_sentiment > -0.05 else "dislike"
    #print(question_sentiment, "sentiment")
    #print(user_input, form_input, noun, orig_noun, noun_topics)
    return form_input, noun, orig_noun, noun_topics
    
    
#find a suitable question template and return it
def find_question_template(processed_text_input):
    max_sim = 0
    max_sim_q = None
    answer_id = 0
        
    max_index, max_sim = cmn.universal_similarity(processed_text_input, likes_embeddings)
    max_sim_q = likes_question_l[max_index]
    answer_id = likes_qid_l[max_index]
    #print(max_sim_q, max_sim, answer_id)
    return answer_id, max_sim, max_sim_q
    
    
    
    
def sent_float_to_text(sentiment):
    ret_sentiment = "love"
    if sentiment < 0.1:
        ret_sentiment = "hate"
    elif sentiment < 0.5:
        ret_sentiment = "dislike"
    elif sentiment < 0.9:
        ret_sentiment = "like"
        
    return ret_sentiment
    
    
    
def fetch_answer_template(answer_id, noun, noun_topics):
    global like_memory
    global question_sentiment
    fetch_answer = template_a.loc[template_a['answer_id'] == answer_id]
    #default
    ans_sentiment = "like"
    ans_sent_val = None
    ret_nouns = noun

    for topic in memory_topics:
        if noun == topic:
            if user_input_sentiment > -0.05: #means the question was neutral or positive
                ret_nouns = topic_favorites[topic]
                ans_sent_val = 0.5
                break
            else: #means the user input question was negative (What food don't you like?)
                ret_nouns = topic_dislike[topic]
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
        #todo... Update:(may already be handled)
        pass
    
    if (((ans_sentiment in sentiment_opt_pos) and question_sentiment in sentiment_opt_pos) 
        or ((ans_sentiment in sentiment_opt_neg) and (question_sentiment in sentiment_opt_neg))):
        fetch_answer = fetch_answer.loc[fetch_answer['same_sentiment'] == 1]
        #if positive sentiment, then clear out any answers which are default negative
        if ans_sentiment in sentiment_opt_pos: 
            fetch_answer = fetch_answer.loc[fetch_answer['default_negative'] != 1]
        #if negative sentiment, clear out default positive answer templates
        else:
            fetch_answer = fetch_answer.loc[fetch_answer['default_positive'] != 1]
       # print("Same sentiment", user_input_sentiment, question_sentiment)
    else:
        fetch_answer = fetch_answer.loc[fetch_answer['same_sentiment'] == 0]
        #print("Not same sentiment",user_input_sentiment, question_sentiment)

        
    #fetch_answer = template_a.loc[template_a['answer_id'] == answer_id ]
    return fetch_answer, ret_nouns, ans_sentiment
    
    
    
    
def process_agent_output(answer_template, noun, nouns, noun_topics, answer_sentiment):
    agent_output = answer_template.answer
    temp_nouns = nouns
    #fetch favorites/least favorites (I forgot how this code works, looks weird)
    if answer_template.fetch_count > 0 and noun_topics != None and len(noun_topics) >0:
        if user_input_sentiment > -0.05: 
            temp_nouns = topic_favorites[noun_topics[0]]
            #like_memory.loc[like_memory['sentiment'] > 0.5 && like_memory['topic'] == noun_topics[0]].sample().subject
        else:
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
    
    
    
    
#------------------Self disclosure component-------------------------------------
def find_user_subject(user_input, question_topic):
    global like_memory
    user_input = user_input.lower()
    user_input_sentiment = cmn.user_sentiment(user_input)
    
    text = cmn.nlp(user_input)
    extracted_nouns = cmn.extract_user_nouns(text)
    
    orig_noun = None
    noun = None
    topic = None
    #"translate" input topic from peilin 
    translated_input_topic, _ = find_similar_noun(question_topic, topic_tokens)
    translated_input_topic = translated_input_topic.lower()
    #check if existing subject
    for n in extracted_nouns:
        if is_noun_existing_subject(n[0]): #checks if noun is in list of subjects
            #get noun topic  
            topic = like_memory.loc[like_memory['lc_subject'] == n[0]].topic.iloc[0]
            topic = topic.lower()
            if topic == translated_input_topic:
                noun = n[0]
                orig_noun = n[1]
                break
    #if not existing subject then check for similar subject
    if noun == None:
        #max_match = None
        max_sim = 0
        for n in extracted_nouns:
            match, sim = find_similar_noun(n[0], subject_tokens)
            if sim > max_sim:
                
                topic = like_memory.loc[like_memory['subject'] == match].topic.iloc[0]
                topic = topic.lower()
                #print(topic,translated_input_topic, n[0],"Match:", match)
                if topic == translated_input_topic:
                    #max_match = match
                    max_sim = sim
                    noun = n[0]
                    orig_noun = n[1]
                    
        if noun != None:
            like_memory = like_memory.append(
                {'subject' : noun , 'topic' : topic, 'sentiment' : max_sim[0], 'lc_subject' : noun.lower()} ,
                ignore_index=True)
    return noun, orig_noun, user_input_sentiment, translated_input_topic


def fetch_disclosure_template(user_subject, user_sentiment):
    #Couldn't identify user input subject, if existed 
    #Therefore Fetch template that says "I like <noun_1>" in the topic
    global disclosure_df
    fetch_df = None
    subj_sent_text = None
    if user_subject == None:
        fetch_df = disclosure_df.loc[disclosure_df['answer_id'] == 2]
    else:
        
        subj_sent_val = fetch_subject_sentiment(user_subject)
        subj_sent_text = sent_float_to_text(subj_sent_val)
        same_sentiment = True
        user_pos = True
        # if user sentiment is positive
        if user_sentiment > -0.05:
            user_pos = True
            fetch_df = disclosure_df.loc[disclosure_df['positive_user'] == user_pos]
            if subj_sent_text in sentiment_opt_pos:
                same_sentiment = True
            else:
                same_sentiment = False
            fetch_df = fetch_df.loc[disclosure_df['same_sentiment'] == same_sentiment]
        else:
            user_pos = False
            fetch_df = disclosure_df.loc[disclosure_df['positive_user'] == user_pos]
    
    return fetch_df, subj_sent_text

          
def disclosure_process_output(template, noun, topic, agent_subject_sentiment):
    disclosure_output = template.answer
    favorite_subjects = None
    if template.fetch_count > 0:
        favorite_subjects = topic_favorites[topic]
    
    #replace nouns
    for i in range(1,template.fetch_count+1):
        temp = "noun_"+str(i)
        disclosure_output = disclosure_output.replace(wildcards[temp], favorite_subjects[i-1])
    
    if template.use_noun:
        disclosure_output = disclosure_output.replace(wildcards["noun"], noun)
    if agent_subject_sentiment != None:
        disclosure_output = disclosure_output.replace(wildcards["agent_sentiment"], agent_subject_sentiment)
    disclosure_output = disclosure_output.replace(wildcards["topic"], topic)
    return disclosure_output   

#---------------------Disclose and reflect component----------------
def disclose_and_reflect(topic):
    translated_input_topic, _ = find_similar_noun(topic, topic_tokens)
    tr_in_topic = translated_input_topic.lower()
    #fetch random template 
    agent_reply = disclose_reflect_df.sample().iloc[0].reply
    
    #get favorite noun in topic 
    fav_subj = topic_favorites[tr_in_topic][0]
    
    fav_sing = fav_subj
    fav_plural = fav_subj 
    topic_sing = singularize(tr_in_topic)
    topic_plural = pluralize(tr_in_topic)
    #hard coded if topic = animal -> pluralize noun 
    if tr_in_topic == "animal":
        fav_sing = singularize(fav_subj)
    
    agent_reply = agent_reply.replace(wildcards["sing_noun"], fav_sing)
    agent_reply = agent_reply.replace(wildcards["plural_noun"], fav_plural)
    agent_reply = agent_reply.replace(wildcards["sing_topic"], topic_sing)
    agent_reply = agent_reply.replace(wildcards["plural_topic"], topic_plural)
    
    return agent_reply

    
    