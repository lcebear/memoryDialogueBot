def fetch_subject_sentiment(key):
    key = key.lower()
    ans_sent = None

    temp_l = like_memory.loc[like_memory['subject'] == key]

    if len(temp_l) > 0:
        ans_sent = temp_l.sentiment.iloc[0]
        #print(temp_l) #if subject listed under multiple 
    #print(key, ans_sent)
    return ans_sent
    
#Input subject to find topic: Apple -> Food/Fruit
def fetch_noun_relations(noun):
    temp_noun_set = set()
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
    temp_l = like_memory.loc[like_memory['subject'] == noun]
    if len(temp_l) > 0:
        return True
    else:
        return False
        
#Check if noun is a known topic in memory
def is_noun_existing_topic(noun):
    return noun in memory_topics
    
    
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

#process the user input 
def process_user_input(user_input):
    extracted_nouns = []
    form_input = user_input
    global question_sentiment
    question_sentiment = "like"
    global like_memory
    noun = None
    sentiment_exist = False
    noun_topics = []
    text = nlp(user_input)
    
    for token in text:
        if token.text in sentiment_opt:
            question_sentiment = token.text
            sentiment_exist = True
            
        tag = nltk.pos_tag([token.text])
        if token.pos_ == "NOUN" or tag[0][1] == "NN" or tag[0][1] == 'NNS':
            extracted_nouns.append(token.lemma_)
            extracted_nouns.append(token.text) 
    for n in extracted_nouns:
        if is_noun_existing_topic(n):
            noun_topics = [noun]
            noun = n
            break
            
        noun_topics = check_noun_topic_exist_memory(n)
        if is_noun_existing_subject(n):
            noun = n
            break
        elif len(noun_topics) >0:
            noun = n
            #if the noun is not a known subject (Apple, Soccer, Pasta) then add it with random sentiment  
            noun_sent = np.random.random(1)
            for topic in noun_topics:
                like_memory = like_memory.append({'subject' : noun , 'topic' : topic, 'sentiment' : noun_sent} , ignore_index=True)
            break
    
    if noun != None:
        if is_noun_existing_topic(noun):
            noun_topics = [noun]
            form_input = form_input.replace(noun, wildcards["topic"])
        else:
            form_input = form_input.replace(noun, wildcards["noun"])
    if sentiment_exist:
        form_input = form_input.replace(question_sentiment, wildcards['sentiment'])

    #print(user_input, form_input, noun)
    return form_input, noun, noun_topics, user_input, extracted_nouns 
    
def find_question_n_answer_retrieval(user_input):
    max_sim = 0
    max_sim_q = None
    answer_id = 0
   
    for i in range(len(retrieval_q)):
        q = retrieval_q.question[i]
        qid = retrieval_q.answer_id[i]
        
        cosine = similarity_calc(q,user_input)

        if int(cosine) == 1:
            max_sim_q = q
            answer_id = qid
            max_sim = cosine
            break
        elif cosine > max_sim:
            max_sim = cosine
            max_sim_q = q
            answer_id = qid
            
    fetch_answer = retrieval_a.loc[retrieval_a['answer_id'] == answer_id]
    answer = fetch_answer.sample().iloc[0].answer
    #print(max_sim_q, max_sim)
    return answer, answer_id, max_sim_q, max_sim
    
#find a suitable question template and return it
def find_question_template(processed_text_input):
    max_sim = 0
    max_sim_q = None
    answer_id = 0
    sent_is_positive = False
    if question_sentiment in sentiment_opt_pos:
        sent_is_positive = True
    for i in range(len(template_q)):
        if sent_is_positive == False and template_q.default_positive[i] == 1:
            continue
        q = template_q.question[i]
        qid = template_q.answer_id[i]
        
        cosine = similarity_calc(q,processed_text_input)

        if int(cosine) == 1:
            max_sim_q = q
            answer_id = qid
            max_sim = cosine
            break
        elif cosine > max_sim:
            max_sim = cosine
            max_sim_q = q
            answer_id = qid
        
    #print(max_sim_q, max_sim)
    return answer_id, max_sim_q, max_sim

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
    
def generate_reply(user_question, num_answers=1):
    text_input = "<|startoftext|>" + user_question
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
    return gen_ans