class Agent():
    def __init__(self):
        
        self.template_q = pd.read_csv('data/likes_question_templates.csv')
        self.retrieval_q = pd.read_csv('data/questions_templates.csv')
        self.template_a = pd.read_csv('data/answer_templates.csv')
        self.retrieval_a = pd.read_csv('data/answer_templates_2.csv')
        self.like_memory = pd.read_csv('data/sentiment_memory.csv')
        
        #Some fruits are listed under "Food" topic, so the line below is temporary remove solution
        self.like_memory = self.like_memory.drop_duplicates(subset='subject', keep="last")
        #Assign random sentiment to every noun item in memory
        temp = np.random.random(len(like_memory))
        self.like_memory['sentiment'] = temp
        
        self.calculate_topic_sent()
    def calculate_topic_sent(self):     
        self.topic_sent = {}
        self.memory_topics = set(like_memory.topic)
        for topic in self.memory_topics:
            topic_list = self.like_memory.loc[like_memory['topic'] == topic]
            count = 0
            divisor = len(topic_list)
            for i in range(divisor):
                count = count + topic_list.sentiment.iloc[i]
            self.topic_sent[topic] = count/divisor
        #print(topic_sent)
     
    def calc_topic_fav(self):
        self.topic_favorites = {}
        select_n = 5 #number of favorites

        for topic in self.memory_topics:
            topic_list = like_memory.loc[like_memory['topic'] == topic]
            topic_list = topic_list.sort_values(by=['sentiment'], ascending=False)
            temp_l = []
            for i in range(select_n):
                temp_l.append(topic_list.subject.iloc[i])#,topic_list.sentiment.iloc[i]))
            topic_favorites[topic] = temp_l
          
        #print(topic_favorites)