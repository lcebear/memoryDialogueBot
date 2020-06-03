import generative_component as gen 

import numpy as np 
import time
from timeit import default_timer as timer

import nltk
from nltk.tokenize import word_tokenize

input_sentence = ''
hist = []
hist_string =""

#Todo, ranking to stay on topic and avoid same questions
#Rank follow up questions high
#Use history embedding
while(1):
    try:
        start = timer()
        hist_string = "<|startofhistory|>"
        for msg in hist:
            hist_string += "<|startoftext|>" + msg + "<|endoftext|>"

        hist_string += "<|endofhistory|>"
        
        
        gen_input = hist_string + "<|startoftext|>"
        #Generate answers
        generated_answers = gen.generate_reply(gen_input)

        for ans in range(len(generated_answers)):
            temp_ans = generated_answers[ans]
            temp_ans = temp_ans.split('?',maxsplit=1)
            generated_answers[ans] = temp_ans[0] + '?'

        print(generated_answers, '\n')

        question = generated_answers[0]
        print(question)
        end = timer()
        print(end-start)
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        
        hist.append(question + " " + input_sentence)


    except Exception as e:
        print(e, "Error: Encountered unknown word.")

