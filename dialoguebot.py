from flask import Flask, request, render_template
import runAgent as ra
app = Flask(__name__) #An instance of class Flask will be our WSGI application.

@app.route('/')
def my_form():
    return render_template('my-form.html')
    
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return str(ra.get_reply(userText)) 
