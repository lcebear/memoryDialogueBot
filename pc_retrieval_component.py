from numpy import save
from numpy import load
import numpy as np
import pandas as pd

import agent_commons as cmn

import time
from timeit import default_timer as timer
print("In pc_retrieval, loading embeddings...")

#TODO: Clean out all questions that exist that don't have question mark
#Otherwise when asked a question, the question will be most similar "document"
pc_retrieval = pd.read_csv('data/pc_retrieval_corpus.csv')

pc_emb =  load('data/pc_embeddings.npy')

retrieval_corp = []

for i in range(len(pc_retrieval)):
    temp_df = pc_retrieval.iloc[i]
    retrieval_corp.append([temp_df.message, temp_df['class']])
 
print("pc_retrieval Loading Complete") 
#TODO Time save exists by only calculating similarity between input and
#The corresponding (translated) answer label instead of all messages in corpus
#Current time est ~0.2 seconds (which is a lot to the already slow system)
def retrieve_similar_message(input_sentence, num_answers=8):
    return fetch_n_most_similar_messages(input_sentence, num_answers=num_answers)

#TODO, can turn input_sentence into an embedding once and send the embedding across
#functions that need it. (How much does it take to turn it into an embedding?
def fetch_n_most_similar_messages(input_sentence, num_answers=8):
    temp_answers = [0] * num_answers
    temp_index = [0] *num_answers
    start = timer()
    #get vectors
    with cmn.g.as_default():
        result = cmn.sim_sess.run(cmn.embedded_text, feed_dict={cmn.text_input: [input_sentence]})
    mid = timer()
    print("Embedding took:", mid -start)
    
    min_index = 0
    min_val = 0
    for i in range(len(pc_emb)):
        sim = np.inner(pc_emb[i],result[0])
        #if a new value is larger than existing minimum
        if sim > min_val:
            #update value at index
            temp_answers[min_index] = sim
            temp_index[min_index] = i
            #temporarily set min val as new entry
            min_val = sim
            #loop through to find the new lowest value (and index)
            for j in range(num_answers):
                if temp_answers[j] < min_val:
                    min_val = temp_answers[j]
                    min_index = j
    for y in range(num_answers):
        temp_answers[y] = retrieval_corp[temp_index[y]][0]
    end = timer()
    print("Final time:", end -start)
    return temp_answers
