import likes_component as disclosure

input_sentence = ''
while(1):
    
    #try:
    # Get input sentence
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit': break
    topic = input('>topic: ')
    noun, orig_noun, user_input_sentiment, translated_topic = disclosure.find_user_subject(input_sentence,topic)
    template, agent_subject_sentiment = disclosure.fetch_disclosure_template(noun, user_input_sentiment)
    if len(template) >0:
        template = template.sample().iloc[0]
    output = disclosure.disclosure_process_output(template, orig_noun, translated_topic, agent_subject_sentiment)
    print(output)
        
    #except KeyError:
     #   print("Error: Encountered unknown word.")

