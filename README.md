# memoryDialogueBot
Master Thesis Project


08/07/2020 - Question answering agent version 2.0. 
Note: The project is not maintained for "plug-n-play".
It may be cleaned and updated at a later date.


The project is developed to be run using the flask library.
Use the following command to set the flask app 'set FLASK_APP = agent_api'.
Then the application can be ran with the following command 'flask run'
The file 'json_post_example' illustrates how the agent can be called through the api.
There is also an interactive version of the agent in file 'run_agent_3',
although the file agent_2.py is the the most up to date, and is used by the flask application.

The latest trained GPT-2 generative model can be downloaded from: https://drive.google.com/open?id=1bamDAUSHM8ye6xzs51-hi9un03DwjOy5
Previous generative version can be downloaded here: https://drive.google.com/open?id=1EeoeK24eIYl44kl3By6KsK8lgcn19qn9
The previous version is still good but it is not specifically trained with history. (does not support '<|startofhistory|>' and '<|endofhistory|>' tokens.

Trained classifiers can be downloaded here: https://drive.google.com/file/d/1coJjE3UL7feooR2x-g6zwedkUPGcKJQA/view?usp=sharing

The project initially used the gpt_2_simple library, but due to concurrent generation problems
and due to the library not being optimized for fast response time (for chatbots), the project now uses
the script "gpt2_specific_gen.py" to generate. 
The script is an adapted variation of "interactive_conditional_samples.py" from openAI and the "gpt_2.py" script from the gpt_2_simple library.


