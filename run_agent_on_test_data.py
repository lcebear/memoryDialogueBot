import agent_2 as agent

import pandas as pd
import ast
#--------------------------
#load the csv with data 
log_data_df = pd.read_csv('data/user-test-1/user_test_data.csv')
#load the survey to get which userID to extract from above dataframe
survey_df = pd.read_csv('data/user-test-1/user_test_survey.csv')

survey_df = survey_df.drop_duplicates(subset='userID', keep="last")
survey_df = survey_df[5:]

userID = survey_df.userID.values
log_data_df = log_data_df[log_data_df['userID'].isin(userID)]

#-
#Fetch conversations
user_dialogue = []
for i in range(len(log_data_df)):
    temp_question_list = []
    messages = log_data_df.iloc[i].messages
    uid = log_data_df.iloc[i].userID
    if uid in userID:
        msg = messages.replace('Timestamp', '')
        msg = ast.literal_eval(msg)
        for m in msg:
            temp_question_list.append(m[0])
        user_dialogue.append(temp_question_list)
#-----------------------------
#Counter to give each conversation a different id (due to how history is used per user)
id_counter = 12000 #Either change this for each test run or clean history csvs

df = pd.DataFrame()

print("Number of conversations:", len(user_dialogue))
temp_dialogue = user_dialogue[0:2]
#loop through every conversation
for convo in user_dialogue:
    id_counter += 1
    print("New convo \n", id_counter)
    if not len(convo) >0 or len(convo)>100:
        continue 
    for i in range(len(convo)):
        answer = "ERROR"
        user_msg = convo[i]
        
        return_json = agent.get_reply(user_msg, id_counter)
        
        if return_json["error"] == None:
            answer = return_json['answer']
        else:
            answer = return_json["error"]
        
        df = df.append({'question' : user_msg, 'answer' : answer,
                       'userID' : id_counter} , ignore_index=True)
            


df.to_csv(r'data/agent_test_data_robustness_check.csv', index = False)



#loop through every message


#call agent with message and id , get reply

#store question with answer 


#end of loops

#save to csv