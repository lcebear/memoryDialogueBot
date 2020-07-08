import pandas as pd
import numpy as np
import agent_commons as cmn 
from generative_component import update_history_embedding, answer_ranking
#Evaluated based on which answer is most relevant or interesting
#0 = No score, 1 = one pair, 2 = 4pair, 3 = Equally good answers
evaluation_score = [0, 0, 0, 0]
model_1_data = pd.read_csv("data/model_selection/m18_test_data_answer.csv")
model_2_data = pd.read_csv("data/model_selection/m19_test_data_answer.csv")
model_3_data = pd.read_csv("data/model_selection/m20_test_data_answer.csv")
model_4_data = pd.read_csv("data/model_selection/m21_test_data_answer.csv")
pastuserID = 0

new_df = pd.DataFrame()
q_hist_emb = None
curr_point = 0
try:
    for i in range(len(model_1_data)):
        #print(i)
        m1 = model_1_data.iloc[i]
        m2 = model_2_data.iloc[i]
        m3 = model_3_data.iloc[i]
        m4 = model_4_data.iloc[i]
        
        if pastuserID != m1.userID:
            print(evaluation_score)
            print("\nNew conversation\n")
            pastuserID = m1.userID
        print() 
        print(m1.question, i)


        print("1.",m1.answer)
        print("2.",m2.answer)
        print("3.",m3.answer)
        print("4.",m4.answer)
        
        #print("0. Skip question")
        #opt = input(">")
        #if opt == "0":
        #    continue
        
        if type(q_hist_emb) is not np.ndarray:
            with cmn.g.as_default():
                q_hist_emb = cmn.sim_sess.run(cmn.embedded_text, feed_dict={cmn.text_input: [m1.question]})
        
        q_hist_emb = update_history_embedding(q_hist_emb, m1.question, alpha=0.2)
        gen_ans = [m1.answer, m2.answer, m3.answer, m4.answer] #[m1.answer, m4.answer]#
        output_sentence = answer_ranking(q_hist_emb, gen_ans, m1.question)
        for j in range(len(gen_ans)): 
            if output_sentence == gen_ans[j]:
                evaluation_score[j] +=1
                curr_point = j
                print()
                print(output_sentence)

        new_df = new_df.append({"question" : m1.question,
        "m18" : m1.answer, "m19" : m2.answer, "m20" : m3.answer, "m21" : m4.answer, "point_to_model" : curr_point }, ignore_index=True)
        
except Exception as e:
    print(e)
finally:
    print(evaluation_score)
    new_df.to_csv('model_selection_automatic_eval_6.csv', index=False)