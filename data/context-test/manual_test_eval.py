import pandas as pd
#Evaluated based on which answer is most relevant or interesting
#0 = Not generated, 1 = one pair, 2 = 4pair, 3 = Equally good answers
evaluation_score = [ 0, 0, 0, 0]
data_onepair_hist = pd.read_csv("ran_on_test_data_true_onepair.csv")
data_4pair_hist = pd.read_csv("ran_on_test_data_4pair.csv")
pastuserID = 0

new_df = pd.DataFrame()
try:
    for i in range(len(data_4pair_hist)):      
        df_4p = data_4pair_hist.iloc[i]
        df_p = data_onepair_hist.iloc[i]
        if pastuserID != df_4p.userID:
            print(evaluation_score)
            print("\nNew conversation\n")
            pastuserID = df_4p.userID
        print() 
        print(df_4p.question)
        if "<GN>" not in df_p.answer:
            score_in = 0
            print(df_p.answer)
            print(df_4p.answer)
        elif df_p.answer == df_4p.answer:
            score_in = 3
            print(df_p.answer)
            print(df_4p.answer)
        else:
            print("1.",df_p.answer)
            print("2.",df_4p.answer) 
            score_in = input('> ')
        if score_in != '1' and score_in != '2' and score_in != 0:
            score_in = 3
        if int(score_in) <= len(evaluation_score):
            evaluation_score[int(score_in)] +=1
        
        new_df = new_df.append({"question" : df_4p.question,
        "p_ans" : df_p.answer, "4p_ans" : df_4p.answer, "point_to_model" : int(score_in) }, ignore_index=True)
        
except Exception as e:
    print(e)
finally:
    print(evaluation_score)
    new_df.to_csv('true_one_vs_4_pair_hist_eval_result.csv', index=False)