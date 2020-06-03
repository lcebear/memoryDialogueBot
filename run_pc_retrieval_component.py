import pc_retrieval_component as pc_corpus

input_sentence = ''
while(1):
    
    try:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        
        answers = pc_corpus.retrieve_similar_message(input_sentence)
        print(answers)
        
    except KeyError:
        print("Error: Encountered unknown word.")

