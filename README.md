# memoryDialogueBot
Master Thesis Project


26/05/2020 - Question answering agent version 2.0. 
The project is developed to be run using the flask library.
Use the following command to set the flask app 'set FLASK_APP = agent_api'.
Then the application can be ran with the following command 'flask run'
The file 'json_post_example' illustrates how the agent can be called through the api.
There is also an interactive version of the agent in file 'run_agent_3',
although the file agent_2.py is the the most up to date, and is used by the flask application.


The latest trained GPT-2 generative model can be downloaded from: https://drive.google.com/open?id=1bamDAUSHM8ye6xzs51-hi9un03DwjOy5
Previous generative version can be downloaded here: https://drive.google.com/open?id=1EeoeK24eIYl44kl3By6KsK8lgcn19qn9
The previous version is still good but it is not specifically trained with history. (does not support '<|startofhistory|>' and '<|endofhistory|>' tokens.

The GPT-2 model does not allow concurrent generation (tensorflow-variable/scope related).
The project's current solution to serve concurrent users is to introduce a queue system.
The same GPT-2 model is loaded multiple times. This is done by duplicating the entire model
into a new folder and running the script
'tensorflow_rename_variables.py' with the command 'py -m tensorflow_rename_variables --checkpoint_dir=”run_18_pc_history_2” --replace_from=scope1 --replace_to=scope1/model --add_prefix=m2/'.
by renaming the variables, we can now load the same model and the models can run in parallel.
Furthermore, gpt_2.py from the gpt_2_simple library is slightly altered to allow running the duplicated
models with different scopes.


