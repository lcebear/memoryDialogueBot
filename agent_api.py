from flask import Flask, request, render_template, session, g, redirect

import agent_2 as agent 

import pandas as pd
import numpy as np

#from timeit import default_timer as timer
#import time

import json

#The generative model seem to return answers within 5 seconds (but has also been observed to take longer)
delay_threshold = 5 

app = Flask(__name__) #An instance of class Flask will be our WSGI application.

#Change the secret key below
app.secret_key = b'\xd6\xedF\xc5\x0f]\x09EF\x06\x0b\xc92\xd5\x94\x96'

@app.route('/', methods=['GET'])
def home():
    return "<h1>MemoryDialogueBot</h1><p>This site is a prototype API for a question answering agent.</p>"



@app.route("/get_answer", methods=['POST'])
def get_bot_response():
    try:
        #gets dictionary 
        req_data = request.get_json()
        return_json = {"Error" : "Unable to access question answering component", "answer" : None}
        if ("userID" in req_data) and ("data" in req_data):
            return_json = agent.get_reply(req_data["data"], req_data["userID"])
        
        
        return return_json
    except Exception as e:
        return {"error" : "Error occured in front-end of API", "answer" : None, "component" : None}

@app.route("/get_disclosure", methods=['POST'])
def get_bot_disclosure():
    try:
        #gets dictionary 
        req_data = request.get_json()
        return_json = {"Error" : "Unable to access self-disclosure component", "answer" : None, "component" : None}
        if ("userID" in req_data) and ("data" in req_data) and ("topic" in req_data):
            return_json = agent.get_self_disclosure(req_data["data"], req_data["userID"], req_data["topic"])
            print(return_json)
            
        return return_json
    except Exception as e:
        return {"error" : "Error occured in front-end of API", "answer" : None, "component" : None}
        
@app.route("/get_disclosure_and_reflect", methods=['POST'])
def get_bot_disclosure_and_reflect():
    try:
        #gets dictionary 
        req_data = request.get_json()
        return_json = {"Error" : "Unable to access disclosure component", "answer" : None, "component" : None}
        if ("userID" in req_data) and ("data" in req_data) and ("topic" in req_data):
            return_json = agent.get_disclose_and_reflect(req_data["data"], req_data["userID"], req_data["topic"])
            print(return_json)

        return return_json
    except Exception as e:
        return {"error" : "Error occured in front-end of API", "answer" : None, "component" : None}

@app.route("/get_question", methods=['POST'])
def get_bot_question():
    try:
        #gets dictionary 
        req_data = request.get_json()
        return_json = {"Error" : "Unable to access question answering component", "answer" : None}
        if ("userID" in req_data) and ("data" in req_data):
            return_json = agent.get_question(req_data["data"], req_data["userID"])
        
        
        return return_json
    except Exception as e:
        return {"error" : "Error occured in front-end of API", "answer" : None, "component" : None}
#------Sleep----------
##ans_split = answer.split()
            
#wps = 2 #240 wpm, absurdly fast typer, average is 40 wpm
##calc = len(ans_split)/wps
#introducin delay mimics human behavior but too long delay may be disruptive even if it's realistic.
#therefore values above threshhold is suppressed using ln 
#if calc > delay_threshold:
#    calc = delay_threshold + np.log(calc) - np.log(delay_threshold)
#Busy wait, thread is blocking and not letting other threads run (no concurrency support)
#end_t = timer()
#print("Took: ",end_t - start_t) # Time in seconds, e.g. 5.38091952400282
#if (end_t-start_t) < calc:
#    print("Calc sec",calc)
#    time.sleep(calc - (end_t-start_t)) #sleep works because each call is a new thread 
#print(threading.currentThread().getName(), 'ending')