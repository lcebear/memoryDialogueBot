import pandas as pd
import os

data_path ="data/"
dir_files = os.listdir(data_path)
file_name = "empty" 
loaded_df = None 
input_sentence = ''
options = ["1. Load csv", "2. Save csv",
"3. print df head", "4. drop rows", "5. drop all rows", "6. drop columns", "7. Drop columns except" ]

def drop_col(df):
    print("Write columns to be dropped, separated by comma")
    print("Example: B,C")
    col_names = input('> ')
    col_names = col_names.split(',')      
    df = df.drop(columns=col_names)
    return df
def keep_col(df):
    print("Write columns to keep, separated by comma")
    print("Example: B,C")
    col_names = input('> ')
    col_names = col_names.split(',')      

    df = df.drop(df.columns.difference(col_names), axis=1)
    return df

def drop_row(df):
    print(df.index)
    print("drop row by index or choose range to keep")
    print("Example index: 342")
    print("Example keep: 0,341")
    row_in = input('> ')
    row_in = row_in.split(',')
    
    if len(row_in)>1:
        df = df.iloc[int(row_in[0]):int(row_in[1])]
    else:
        df = df.drop(df.index[int(row_in[0])])
    
    #df.drop(columns=['B', 'C'])
    return df
    

while(1):
    try:
        for opt in options:
            print(opt)
        # Get input sentence
        input_sentence = ''
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        
        if input_sentence == '1':
            for i in range(len(dir_files)):
                print("\t", str(i) + ".", dir_files[i])
            file_name = input('> ')
            
            loaded_df = pd.read_csv('data/' + dir_files[int(file_name)]) 
        elif input_sentence == '2':
            assert(isinstance(loaded_df, pd.DataFrame))
            loaded_df.to_csv('data/' + dir_files[int(file_name)], index=False)
        elif input_sentence == '3':
            assert(isinstance(loaded_df, pd.DataFrame))
            print(loaded_df)
        elif input_sentence == '4':
            assert(isinstance(loaded_df, pd.DataFrame))
            loaded_df = drop_row(loaded_df)
        elif input_sentence == '5':
            assert(isinstance(loaded_df, pd.DataFrame))
            loaded_df = loaded_df.iloc[0:0]
        elif input_sentence =='6':
            assert(isinstance(loaded_df, pd.DataFrame))
            loaded_df = drop_col(loaded_df)
        elif input_sentence =='7':
            assert(isinstance(loaded_df, pd.DataFrame))
            loaded_df = keep_col(loaded_df)
            


    except Exception as e:
        print("Error:", e)