# memoryDialogueBot
Master Thesis Project


07/09/2020 - Question answering agent version 2.0. 

The project (API) is developed to be run using the flask and waitress library.
The application can be ran with the following command 'python3 waitress_server.py'
The file 'json_post_example.py' illustrates how the agent can be called through the api.
There is also an interactive version of the agent in file 'run_agent.py'.

The latest trained GPT-2 generative model can be downloaded from: https://drive.google.com/open?id=1bamDAUSHM8ye6xzs51-hi9un03DwjOy5

Trained classifiers can be downloaded here: https://drive.google.com/file/d/1coJjE3UL7feooR2x-g6zwedkUPGcKJQA/view?usp=sharing

The project initially used the gpt_2_simple library, but due to concurrent generation problems
and due to the library not being optimized for fast response time (for chatbots), the project now uses
the script "gpt2_specific_gen.py" to generate. 
The script is an adapted variation of "interactive_conditional_samples.py" from OpenAI and the "gpt_2.py" script from the gpt_2_simple library.


# Setup
```
git clone https://github.com/lcebear/memoryDialogueBot.git
```
Create a virtual python environment
```
python3 venv path/to/env
```
Install requirements
```
pip install -r requirements.txt
```
Download additional packages
```
python -m spacy download en_core_web_lg
```
download nltk packages 'stopwords', 'punkt', 'vader_lexicon', 'averaged_perceptron_tagger' by using nltk.download()
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download(averaged_perceptron_tagger')
...
```
[Download](https://drive.google.com/open?id=1bamDAUSHM8ye6xzs51-hi9un03DwjOy5) latest GPT-2 generative model to project folder.

[Download](https://drive.google.com/file/d/1coJjE3UL7feooR2x-g6zwedkUPGcKJQA/view?usp=sharing) trained classifiers to project folder.

## Run Chatbot
```
python3 -m run_agent.py 
```
## Run API
```
python3 waitress_server.py
```
-> Example API usage in json_post_example.py

Remember to change the secret key (app.secret_key) in the 'agent_api' file to some random bytes.

# Cite
TBD - Master Thesis still on-going (report)
