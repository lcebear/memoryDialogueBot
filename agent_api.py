from flask import Flask, request, render_template, session, g, redirect

import agent_2 as agent 

import pandas as pd
import numpy as np

from timeit import default_timer as timer
import time

import threading
import json



app = Flask(__name__) #An instance of class Flask will be our WSGI application.

#Change the secret key below
app.secret_key = b'\xd5\xedF\xc5\x0f]\x08EF\x06\x0b\xc9l\xd5\x94\x95'

@app.route('/', methods=['GET'])
def home():
    return "<h1>MemoryDialogueBot</h1><p>This site is a prototype API for a question answering agent.</p>"



@app.route("/get_answer", methods=['POST'])
def get_bot_response():
    #gets dictionary 
    req_data = request.get_json()
    return_json = {"Error" : "Unable to access question answering component", "answer" : None}
    if ("userID" in req_data) and ("data" in req_data):
        return_json = agent.get_reply(req_data["data"], req_data["userID"])
    
    return return_json


# #----------------------------------------
# global idvar 
# idvar = 9999

# data_lock = threading.Lock()

# user_test_frame = pd.read_csv('data/user_test_id.csv')
# user_test_data_frame = pd.read_csv('data/user_test_data.csv')
# user_survey_frame = pd.read_csv('data/user_test_survey.csv')

# idvar = int(user_test_frame['userID'].max())
# print(idvar)
# message_history = {}

# delay_threshhold = 8
# #stats
# unique_visits = 0
# num_completed = 0


# @app.route('/', )
# def about():
    # global user_test_frame, unique_visits
    # print("Home- Session id:", session.get('userID'))
    
    # session['maxDelay'] = 0
    # session['avgDelay'] = 0
    # #if the user has an id then pass this if-else, else give user an ID
    # if session.get('userID') != None:
        # pass
    # else:
        # data_lock.acquire()
        # try:
            # unique_visits +=1
            # global idvar
            # idvar +=1
            # session['userID'] = idvar
            # user_test_frame = user_test_frame.append({'userID' : session['userID'] , 'time' : pd.Timestamp.now() } , ignore_index=True)
            # user_test_frame.to_csv(r'data/user_test_id.csv', index = False)
            # print("set ID")
            
        # finally:
            # data_lock.release()
            
        # ra.agent.df_lock.acquire()
        # try:
            # ra.agent.user_history = ra.agent.user_history.append({'userID' : session['userID'] , 'message_history' : [], 'true_sentiment' : [] } , ignore_index=True)
            # print("Added ", session['userID'], "to user history")
        # except Exception as e:
            # print(e, "tryed to add userID to user history")
        # finally:
            # ra.agent.df_lock.release()
    
    # session['delay'] = True if(session.get('userID') % 2) else False
    # #g.value = idvar
    
    # return render_template('about.html')
    

# @app.route('/chat', methods=['GET', 'POST'])
# def my_form():
    # global message_history
    # if session.get('userID') != None:
        # print("Chat- Session id:", session.get('userID'))
    # else:
        # print("no session, redirecting")
        # return redirect("/")
    # if request.method == 'POST':
        # print("OI, in chat")
        # #temp_df = user_test_data_frame.loc[user_test_data_frame['userID'] == session['userID']]
        # #if len(temp_df)>0:
        # #    message_history[session['userID']] = temp_df.messages.iloc[0]
        # #else:
        # message_history[session['userID']] = []

    # elif request.method == 'GET':
        # print("method= GET, on page /chat")
        # return redirect("/")

    # return render_template('chat.html')

# #called in <script> in my-form.html    
# @app.route("/get_gen")
# def get_bot_response():
    # start_t = timer()
    
    # global message_history
    # temp_l = message_history[session['userID']]
    
    # #save messages (user input and bot output)
    # userText = request.args.get('msg')
    # user_msg_time = pd.Timestamp.now().round('s')
    # reply = ra.get_reply(userText, session['userID'])


    
    # #artificial dynamic delay
    # if session.get('delay') == True:    
        
        # answer = reply['answer']
        # ans_split = answer.split()
        
        # wps = 2 #240 wpm, absurdly fast typer, average is 40 wpm
        # calc = len(ans_split)/wps
        # #introducin delay mimics human behavior but too long delay may be disruptive even if it's realistic.
        # #therefore values above threshhold is suppressed using ln 
        # if calc > delay_threshhold:
            # calc = delay_threshhold + np.log(calc) - 2
        # #Busy wait, thread is blocking and not letting other threads run (no concurrency support)
        # end_t = timer()
        # print("Took: ",end_t - start_t) # Time in seconds, e.g. 5.38091952400282
        # if (end_t-start_t) < calc:
            # print("Calc sec",calc)
            # time.sleep(calc - (end_t-start_t))
            
    # bot_msg_time = pd.Timestamp.now().round('s')    
    # temp_l.append((userText,reply,user_msg_time, bot_msg_time))
    
    # #get number of messages before this one.
    # n_msg = len(message_history[session['userID']])
    
    # #update dictionary of messages for this user 
    # message_history[session['userID']] = temp_l  
    # curr_t = timer()
    # temp_diff_time = curr_t - start_t
    
    # #calculate and update mean delay 
    # mean_delay = session['avgDelay']*n_msg 
    # mean_delay = (mean_delay+ temp_diff_time)/(n_msg+1)
    # session['avgDelay'] = mean_delay
    
    # #update max delay 
    # if temp_diff_time > session.get('maxDelay'):
        # session['maxDelay'] = temp_diff_time
    
    # return reply['answer']

# @app.route('/survey', methods=['GET', 'POST'])
# def survey():
    # global user_test_data_frame
    # if request.method == 'POST':
        # print("OI, in survey")
        
        # #save conversation
        # #temp_df = user_test_data_frame.loc[user_test_data_frame['userID'] == session['userID']]
        # #if len(temp_df)>0:
        # session['userTime'] = pd.Timestamp.now().round('s')
        # data_lock.acquire()
        # try:
            # user_test_data_frame = user_test_data_frame.append({'userID' : session['userID'] , 'time' : session['userTime'],
            # 'messages' : message_history[session['userID']], 'num_messages' : len(message_history[session['userID']]) } , ignore_index=True)
            # user_test_data_frame.to_csv(r'data/user_test_data.csv', index = False)
        # finally:
            # data_lock.release()
            
        # ra.agent.df_lock.acquire()
        # try:
            # ra.agent.user_history.to_csv(r'data/user_history.csv', index = False)
            # print("Saving generated_answers_kb")
            # ra.agent.generated_kb.to_csv(r'data/generated_answers_kb.csv', index = False)
        # finally:
            # ra.agent.df_lock.release()
 
        # return render_template('survey.html')
    # else:
        # return redirect("/")
    
# @app.route('/end', methods=['GET', 'POST'])
# def finished_survey():
    # global user_survey_frame, num_completed
    # if request.method == 'POST':
    
        # survey_dict ={
        # 'age' : request.form['age'],
        # 'english-proficiency' : request.form['english'],
        # 'previousChatbots' : request.form['previousChatbots'],
        # 'engagingness' : request.form['engagingness'],
        # 'unresponsiveness' : request.form['unresponsiveness'], 
        # 'realness' : request.form['realness'],
        # 'inconsistency' : request.form['inconsistency'],
        # 'relevancy' : request.form['relevancy'],
        # 'repetitiveness' : request.form['repetitiveness'],
        # 'responsiveness' : request.form['responsiveness'],
        # 'retention' : request.form['retention'],
        # 'persona' : request.form['persona'] }
        # data_lock.acquire()
        # try:
            # num_completed +=1
            # user_survey_frame = user_survey_frame.append({'userID' : session['userID'] , 'time' : session['userTime'],
            # 'survey' : survey_dict, 'delay' : session['delay'], 'max_delay' : session.get('maxDelay'), 'avg_delay' : session['avgDelay'] } , ignore_index=True)
            # user_survey_frame.to_csv(r'data/user_test_survey.csv', index = False)
            # print("OI, in end")
        # finally:
            # data_lock.release()
        
        # #TODO save survey+messages+id+time
    # return render_template('end.html')

# @app.route('/livestats')
# def check_stats():
    # return render_template('livestats.html', visits=unique_visits, completed=num_completed)