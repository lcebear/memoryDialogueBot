import numpy as np
import tensorflow as tf
import requests

#NLP
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pattern.en import pluralize, singularize


#universal sentence encoder
import tensorflow_hub as hub

import pickle
from sklearn.neighbors import KNeighborsClassifier 

#Load spacy
nlp = spacy.load('en_core_web_lg')
#Vader sentiment
sid = SentimentIntensityAnalyzer()
#Universal sentence encoder
# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.compat.v1.global_variables_initializer(),
                      tf.compat.v1.tables_initializer()])
g.finalize()
# Create session and initialize.
sim_sess = tf.compat.v1.Session(graph=g)
sim_sess.run(init_op)

#Calculate similarity measure between input sentence and a list of embeddings
#Universal sentence encoder + cosine similarity
def universal_similarity(user_in, in_emb):
    max_sim = 0
    max_index = 0
    with g.as_default():
        result = sim_sess.run(embedded_text, feed_dict={text_input: [user_in]})
    for i in range(len(in_emb)):
            sim = np.inner(in_emb[i],result[0])
            if sim > max_sim:
                max_sim = sim
                max_index = i

    return max_index, max_sim
    
#VADER sentiment   
def user_sentiment(sentence):
    ss = sid.polarity_scores(sentence)
    return ss['compound']
    
#Input spacy tokens
def extract_user_nouns(user_tokens):
    prev_token = None 
    temp_str =""
    extracted_nouns = []
    
    for token in user_tokens:
        #Nltk pos tag    
        tag = nltk.pos_tag([token.text])
        if token.pos_ == "NOUN" or tag[0][1] == "NN" or tag[0][1] == 'NNS':
            #Enabling double noun words "Video games" "TV shows"
            if prev_token != None:
                temp_str = prev_token + " " + token.text
                extracted_nouns.insert(0, (temp_str, temp_str)) #inserted first in the list
            
            extracted_nouns.append((token.text, token.text))
            extracted_nouns.append((token.lemma_, token.text))
            extracted_nouns.append((pluralize(token.text), token.text))
        prev_token = token.text if token.pos_ == "NOUN" else None #could be extended for nltk tag
    return extracted_nouns

 
    
#----------------------------Classifier commons-----------------------------------    

#load classifier model from disk
loaded_knn_question = pickle.load(open(r'classifier/knn_question_classifier', 'rb'))
loaded_knn_answer = pickle.load(open(r'classifier/knn_answer_classifier', 'rb'))
loaded_qna = pickle.load(open(r'classifier/knn_qna_classifier', 'rb'))

question_labels = {0 : "hobbies/interests", 1 : "where are you from?",
                   2 : "kids/married/pets/male/female",
                  3 : "Follow up question type", 4 : "Job",
                  7 : "Movies/series/reading", 8 : "sports/Games",
                   9 : "Follow up question type", 10 : "How long...?", 11 : "And you? (reflect question)",
                  12 : "Food/Drink", 13 : "Weather", 14 : "How are you?", 15 : "What's your name?",
                  16 : "Student/studies", 17 : "music/instrument", 18 : "Really?/surprised reaction",
                  19 : "Traveling", 20 : "'Awesome'/'cool'", 22 : "understandable/interesting",
                  24 : "Yes/no/indeed", 27 : "Thank you", 29 : "Hi / Hello",
                  30 : "Good luck/congrats", 33 : "Me too/same here", 34 : "My name is",
                  36 : "Nice to meet you", 70 : "Good morning/ i'm good", 100 : "LOL, hahaha"}

answer_labels = {0 : "muscle/car/biking", 2 : "Location/residence",
                   3 : "Likes/plans/outdoor", 4 : "reading/studying",
                  6 : "Movies/series", 7 : "Good morning/ i'm good",
                  9 : "Travel/Language", 10 : "LOL, hahaha", 12 : "work",
                   13 : "years/numbers", 16 : "dance/music", 18 : "Hobbies/interests",
                  19 : "Study/Work", 20 : "'Awesome'/'cool'", 22 : "understandable/interesting",
                 24 : "Yes/no/indeed", 25 : "Pets", 26 : "food/cooking",
                 27 : "Thank you", 29 : "Hi / Hello", 30 : "Good luck/congrats",
                  31 : "Age/social status/children", 32 : "sports teams",
                 33 : "Me too/same here", 34 : "My name is",
                  35 : " Weather", 37 : "money/work", 39 : "Study/student/teacher/classes"}

#translate labels
#non_translatable
non_translatable_labels = [3, 11, 9, 20, 22, 24, 27, 29, 30, 33, 34, 36, 70, 100]
translate_dict = {0 : [18], 1 : [2,9], 2 : [25,31], 4 : [12,19], 7 : [6,4],
                 8 : [32, 18, 3], 10 : [13], 12 : [26], 13 : [35], 14 : [7], 15 : [34],
                  16 : [39,4,19], 17 : [16], 18 : [24], 19 : [9] }
#Translate question label to matching answer label, given question label, return possible answer labels
#answer_labels = 
#print(answer_labels[translate_dict[1][0]])
def classify_qna(user_inp):

    with g.as_default():
        emb = sim_sess.run(embedded_text, feed_dict={text_input: [user_inp]})
    pred = loaded_qna.predict(emb)
    return pred[0]


def classify_question(user_q):
    with g.as_default():
        q_emb = sim_sess.run(embedded_text, feed_dict={text_input: [user_q]})
        
    pred_q = loaded_knn_question.predict(q_emb)
    if pred_q[0] in translate_dict:
        t_id = translate_dict[pred_q[0]]
    else:
        t_id = None
    #print(question_labels[pred_q[0]], pred_q[0], t_id)
    return pred_q[0]
        
def classify_answers(agent_ans):
    with g.as_default():
        ans_emb = sim_sess.run(embedded_text, feed_dict={text_input: agent_ans})
        
    pred_ans = loaded_knn_answer.predict(ans_emb)
    return pred_ans
    
    
def print_answer_labels(ans,ans_labels):
    for i in range(len(ans)):
        print("Answer:", ans[i], "Label:", answer_labels[ans_labels[i]], "ID:", ans_labels[i])

  
#-----------------------------------------------------------------------