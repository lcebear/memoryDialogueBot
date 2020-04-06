from flask import Flask, request, render_template, session, g, redirect
import runAgent as ra
import pandas as pd
from timeit import default_timer as timer
import time


app = Flask(__name__) #An instance of class Flask will be our WSGI application.
app.secret_key = b'\xd5\xedF\xc5\x0f]\x08EF\x06\x0b\xc9l\xd5\x94\x95'
global idvar 
idvar = 9999

user_test_frame = pd.read_csv('data/user_test_id.csv')
user_test_data_frame = pd.read_csv('data/user_test_data.csv')
user_survey_frame = pd.read_csv('data/user_test_survey.csv')
idvar = int(user_test_frame['userID'].max())
print(idvar)
message_history = {}
# Wait for 5 seconds


@app.route('/', )
def about():
    global user_test_frame
    print("Home- Session id:", session.get('userID'))
    if session.get('userID') != None:
        pass
    else:
        global idvar
        idvar +=1
        session['userID'] = idvar
        user_test_frame = user_test_frame.append({'userID' : session['userID'] , 'time' : pd.Timestamp.now() } , ignore_index=True)
        user_test_frame.to_csv(r'data/user_test_id.csv', index = False)
        print("set ID")
    
    
    session['delay'] = True if(session.get('userID') % 2) else False
    #g.value = idvar
    
    return render_template('about.html')
    

@app.route('/chat', methods=['GET', 'POST'])
def my_form():
    global message_history
    if session.get('userID') != None:
        print("Chat- Session id:", session.get('userID'))
    else:
        print("no session, redirecting")
        return redirect("/")
    if request.method == 'POST':
        print("OI, in chat")
        #temp_df = user_test_data_frame.loc[user_test_data_frame['userID'] == session['userID']]
        #if len(temp_df)>0:
        #    message_history[session['userID']] = temp_df.messages.iloc[0]
        #else:
        message_history[session['userID']] = []

    elif request.method == 'GET':
        print("method= GET, on page /chat")
        return redirect("/")

    return render_template('chat.html')

#called in <script> in my-form.html    
@app.route("/get_gen")
def get_bot_response():
    start_t = timer()
    
    global message_history
    temp_l = message_history[session['userID']]
    
    #TODO save messages (user input and bot output)
    userText = request.args.get('msg') 
    reply = ra.get_reply(userText)
    temp_l.append((userText,reply))
    message_history[session['userID']] = temp_l
    #print(message_history[session['userID']])
    
    #artificial dynamic delay
    if session.get('delay') == True:    
        
        answer = reply['answer']
        ans_split = answer.split()
        #print(len(ans_split))
        wps = 4 #240 wpm, absurdly fast typer, average is 40 wpm
        calc = len(ans_split)/wps
        #Busy wait, thread is blocking and not letting other threads run (no concurrency support)
        end_t = timer()
        print("Took: ",end_t - start_t) # Time in seconds, e.g. 5.38091952400282
        if (end_t-start_t) < calc:
            print("Calc sec",calc)
            time.sleep(calc - (end_t-start_t))
        
        
    return reply['answer']

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    global user_test_data_frame
    if request.method == 'POST':
        print("OI, in survey")
        
        #save conversation
        #temp_df = user_test_data_frame.loc[user_test_data_frame['userID'] == session['userID']]
        #if len(temp_df)>0:
        session['userTime'] = pd.Timestamp.now().round('s')
        user_test_data_frame = user_test_data_frame.append({'userID' : session['userID'] , 'time' : session['userTime'],
        'messages' : message_history[session['userID']], 'num_messages' : len(message_history[session['userID']]) } , ignore_index=True)
        user_test_data_frame.to_csv(r'data/user_test_data.csv', index = False)
        print("Saving generated_answers_kb")
        ra.agent.generated_kb.to_csv(r'data/generated_answers_kb.csv', index = False)
 
    return render_template('survey.html')
    
@app.route('/end', methods=['GET', 'POST'])
def finished_survey():
    global user_survey_frame
    if request.method == 'POST':
    
        survey_dict ={
        'previousChatbots' : request.form['previousChatbots'],
        'age' : request.form['age'],
        'engagingness' : request.form['engagingness'],
        'unresponsiveness' : request.form['unresponsiveness'], 
        'realness' : request.form['realness'],
        'inconsistency' : request.form['inconsistency'],
        'relevancy' : request.form['relevancy'],
        'repetitiveness' : request.form['repetitiveness'],
        'responsiveness' : request.form['responsiveness'],
        'retention' : request.form['retention'],
        'persona' : request.form['persona'] }
        
        user_survey_frame = user_survey_frame.append({'userID' : session['userID'] , 'time' : session['userTime'], 'survey' : survey_dict, 'delay' : session['delay'] } , ignore_index=True)
        user_survey_frame.to_csv(r'data/user_test_survey.csv', index = False)
        print("OI, in end")
        
        #TODO save survey+messages+id+time
    return render_template('end.html')

