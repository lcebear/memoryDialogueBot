from flask import Flask, request, render_template
import runAgent as ra
app = Flask(__name__) #An instance of class Flask will be our WSGI application.

@app.route('/', )
def about():
    return render_template('about.html')
    

@app.route('/chat', methods=['GET', 'POST'])
def my_form():
    if request.method == 'POST':
        print("OI, in chat")
        #TODO: Get user id here and initialize session
    elif request.method == 'GET':
        print("method= GET, on page /chat")

    return render_template('chat.html')

#called in <script> in my-form.html    
@app.route("/get_gen")
def get_bot_response():
    #TODO save messages (user input and bot output)
    userText = request.args.get('msg')    
    return str(ra.get_reply(userText)) 

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        print("OI, in survey")
        #TODO save generated questions and answers, (save conversation?)
    return render_template('survey.html')
    
@app.route('/end', methods=['GET', 'POST'])
def finished_survey():
    if request.method == 'POST':
        print("OI, in end")
        #TODO save survey+messages+id+time
    return render_template('end.html')

