import likes_component as likes

input_sentence = ''
while(1):
    
    #try:
    # Get input sentence
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit': break
    answer = likes.disclose_and_reflect(input_sentence)
    print(answer)  
    #except KeyError:
     #   print("Error: Encountered unknown word.")

