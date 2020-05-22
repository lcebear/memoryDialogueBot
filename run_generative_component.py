import generative_component as gen 

import numpy as np 
import time
from timeit import default_timer as timer

import nltk
from nltk.tokenize import word_tokenize

input_sentence = ''
qa_history_embedding = None
msg_hist = []
hist = ""
#2 == 1 question and answer pair
num_hist_msg = 2

answer_past_label = 0
question_past_label = 0

question_string = ""
# 3,9,10 follow up, whereas 18, 20, 22, 24,27, 30, 33 are generic
follow_up_q = [3,9, 10, 18, 20, 22, 24,27, 30, 33]
past_answer = None

is_follow_up_question = False
while(1):
    #try:
    # Get input sentence
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit': break
    start = timer()
    q_label = gen.cmn.classify_question(input_sentence)
    
    
    with gen.cmn.g.as_default():
        q_emb = gen.cmn.sim_sess.run(gen.cmn.embedded_text, feed_dict={gen.cmn.text_input: [input_sentence]})
        
    #pred_q = gen.cmn.loaded_knn_question.predict_proba(q_emb)
    #print(pred_q)
    
    
    #Runs once, for the first sentence.
    if type(qa_history_embedding) is not np.ndarray:
        with gen.cmn.g.as_default():
            qa_history_embedding = gen.cmn.sim_sess.run(gen.cmn.embedded_text, feed_dict={gen.cmn.text_input: [input_sentence]})
        question_past_label = q_label
    
    hist_q_lab = gen.cmn.loaded_knn_question.predict(qa_history_embedding)
    hist_a_lab = gen.cmn.loaded_knn_answer.predict(qa_history_embedding)
    print(hist_q_lab[0], gen.cmn.question_labels[hist_q_lab[0]] ) 
    print(hist_a_lab[0], gen.cmn.answer_labels[hist_a_lab[0]])

    question_string ="<|startoftext|>"+input_sentence

    #adding question mark if sentence doesn't have it
    tokens = word_tokenize(input_sentence)
    if nltk.tag.pos_tag([tokens[-1]])[0][1] != '.':
        question_string = question_string + "?"


    
    # if len(msg_hist)> num_hist_msg:
        # hist = "<|startofhistory|>"
        # for y in range(num_hist_msg):
            # x = num_hist_msg -y
            # hist += msg_hist[-x] +'\n'
        # hist += "<|endofhistory|>"
        # gen_input = hist + question_string
    # else:
        # gen_input = "<|startofhistory|>" + hist + "<|endofhistory|>" + question_string
    
    # msg_hist.append(question_string)
    
    hist = "<|startofhistory|>" + "<|endofhistory|>"
    if len(msg_hist) > 1:
        if (q_label in follow_up_q) or q_label == question_past_label:
            is_follow_up_question = True
            hist = "<|startofhistory|>" + msg_hist[-2] + msg_hist[-1] + "<|endofhistory|>"
        else:
            is_follow_up_question = False
            
    gen_input = hist + question_string
    
    msg_hist.append(question_string)


    print(gen_input)

    #gen_input ="<|startoftext|>"+input_sentence 
    
    #If follow up question, use higher alpha to remain previous history embedding(the answer)
    if is_follow_up_question:
        qa_history_embedding = gen.update_history_embedding(qa_history_embedding, input_sentence, alpha=0.4)
    else:
        qa_history_embedding = gen.update_history_embedding(qa_history_embedding, input_sentence, alpha=0.2)
    
    #Generate answers
    generated_answers = gen.generate_reply(gen_input)
    #remove bad answers (empty or too similar to past answer)
    generated_answers = gen.bad_answer_removal(generated_answers, past_answer)
    #Run the answers through a knn answer classifier 
    classified_generated_answers = gen.cmn.classify_answers(generated_answers)
    
    #print answers and their labels
    gen.cmn.print_answer_labels(generated_answers, classified_generated_answers)
    
    #rank answers
    output_sentence = gen.answer_ranking(qa_history_embedding, generated_answers, input_sentence)
    
    #output_sentence = ?
    
    #output_sentence, answer_past_label = gen.generate_reply(qa_history_embedding, gen_input, q_label,question_past_label, answer_past_label,input_sentence, past_answer)

    qa_history_embedding = gen.update_history_embedding(qa_history_embedding, output_sentence)
    print(output_sentence, '\n')
    #print(ans, '\n')
    end = timer()
    print(end - start)
    past_answer = output_sentence
    answer_string = " " + output_sentence + '<|endoftext|>'
    msg_hist.append(answer_string)
    hist += question_string + answer_string
    question_past_label = q_label
    #except Exception as e:
     #   print(e, "Error: Encountered unknown word.")
        