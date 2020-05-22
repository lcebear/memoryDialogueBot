import likes_component as likes 

print("Agent Successfully loaded\n")

input_sentence = ''
while(1):
    try:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break

        processed_text_input, user_noun, orig_noun, noun_topics  = likes.process_user_input(input_sentence)
        if user_noun != None:
            #print(user_noun, orig_noun)
            #print(user_input_sentiment, question_sentiment)
            answer_id, max_sim_val = likes.find_question_template(processed_text_input)[0:2]

            answer, nouns, answer_sentiment = likes.fetch_answer_template(answer_id, user_noun, noun_topics)
            if len(answer) >0:
                answer = answer.sample().iloc[0]
            else:
                print("I don't know", nouns,answer_id, answer_sentiment)
            #print(answer_id)
            print(likes.process_agent_output(answer,
                                             orig_noun, nouns,noun_topics, answer_sentiment))
        else:
            print("I don't know.")
        
    except KeyError:
        print("Error: Encountered unknown word.")