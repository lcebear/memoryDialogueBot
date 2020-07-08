import agent_2 as agent

import pandas as pd
import ast

input_quests = """what do you do for fun?
What do you do for a living?
What do you like to do in your spare time?
what else do you like doing?
What are you studying?
what do you do for fun in Missouri?
Do you love American food?
Are you ready for the snow to melt and spring to start?
Are you a student at Harvard?
What kind of education do you have?
Where do you live?
What type of job do you do?
Have you seen any good movies yet?
Do you cook?
What do you do with your summer?
What kind of music do you like?
What's your favorite cake you have made?
do you have any recommendations for things to do in the area?
do you like The Count from Sesame Street?
how often do you do these types of studies?
do you have any pets?
Are you into yoga?
What kind of field do you study in?
What do you do?
Do you like reading?
So are you a student or working?
Any favourite sports?
Have you ever seen snow?
How is the weather where you are?
Do you like to watch any sports?
do you like to travel?
do you have any hobbies?
do you have plans for the weekend?
are you from here?
Did you study abroad at all or plan to?
What's your major?
What's your opinion on him?
Do you go to school in Boston?
You really like to read, don't you?
When did you start singing?
Do you have children?
How old are your kids?
How's your morning going?
Do you want to stay in the area or move to another state?
Do you have any plans for the weekend?
do you have a favorite kind of music you like to listen to?
Have you always lived in Texas?
How do you like playing sports?
do you like soccer?
Have you seen ANY of the Hunger games movies?
Are you old enough to drink?
Do you ski?
How are you today?
do you meditate?
What's your favorite beer?
Are you on Facebook?
Are you also in college?
have you ever been to italy?
Do you speak any other languages?
What's your favorite kind of food?
What is your favorite place in the entire country?
what is your name?
What do you do outside of school?
What kind of ice cream flavor would you suggest for me?
what sport do you play?
are you religious?
you like anything to do with computers?
What's a typical day like for you?
Have you enjoyed the sunshine?
is there anything fun planned for you this weekend?
what do you do that is important to you in your life?
Can I ask your age?
Do you enjoy reading?
What are you planning to do on your vacation?
Have you ever been in Asia?
do you enjoy swimming?
What do you do at home on a Saturday?
What are your favorite genres then?
What do you do when stressed?
Are you an artist?
What do you do when you're on the computer?
What's your favorite thing to cook?
where in the world do you live?
Are you a monk too?
What do you do with your winter vacation?
Did you ever leave the country before? """
input_quests = input_quests.split("\n")

#-----------------------------
#Counter to give each conversation a different id (due to how history is used per user)
user_id = "m21_test_run" #Either change this for each test run or clean history csvs

df = pd.DataFrame()



for q in input_quests:
    answer = "ERROR"
    
    return_json = agent.get_reply(q, user_id, only_generated=True)
    
    if return_json["error"] == None:
        answer = return_json['answer']
    else:
        answer = return_json["error"]
    
    df = df.append({'question' : q, 'answer' : answer,
                   'userID' : user_id} , ignore_index=True)
            


df.to_csv(r'data/m21_test_data_answer.csv', index = False)



#loop through every message


#call agent with message and id , get reply

#store question with answer 


#end of loops

#save to csv