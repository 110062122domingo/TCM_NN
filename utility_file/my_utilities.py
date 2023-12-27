import json
import os
import sys
import pandas as pd
from tabulate import tabulate

def dict_to_txt(dictToPrint: dict,  save_path:str=None, file_prefix: str='', textbox: str=None):
        if save_path is None:
            save_path='./result/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        add_num =1
        if len(file_prefix) ==0:
             file_prefix = "accuracy"
        save_path += f"/{file_prefix}"
        save_path_norepeat = save_path
        while os.path.isfile(f'{save_path_norepeat}.txt'):
            save_path_norepeat = save_path + f'_{add_num}'
            add_num +=1

        save_path_norepeat += '.txt'

        with open(save_path_norepeat, 'w', encoding='utf-8') as file:
            # try:
            #     json.dumps(dictToPrint)
            #     json.dump(dictToPrint, file, indent=2)
                
            # except:
            #     print("cannot dump dictToPrint, use normal print")
            
            #     old_stdout = sys.stdout
            #     sys.stdout = file
            #     print(dictToPrint)
            #     sys.stdout = old_stdout  # Reset stdout to default (console)
            # 
            old_stdout = sys.stdout
            sys.stdout = file
            print(dictToPrint)
            sys.stdout = old_stdout  # Reset stdout to default (console) 
            # can't use sys.stdout = sys.__stdout__  in jupyter notebook!!! will cause error!!
            try:
                if textbox is not None:
                    file.write('\n')
                    file.write(textbox)
                    file.write('\n')
            except:
                pass
        print(f"{file_prefix} saved to {save_path_norepeat}")

def df_to_csv(df: pd.DataFrame, save_path:str=None, file_prefix: str='', textbox: str=None):
        """
        export df to csv in dir {save_path} with auto file naming
        """
        if save_path is None:
            save_path='./result/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        add_num =1
        if len(file_prefix) ==0:
             file_prefix = "predictions"
        save_path += f"/{file_prefix}"
        save_path_norepeat = save_path
        while os.path.isfile(f'{save_path_norepeat}.csv'):
            save_path_norepeat = save_path + f'_{add_num}'
            add_num +=1

        save_path_norepeat += '.csv'
        df.to_csv(save_path_norepeat)
        if textbox is not None:
            with open(save_path_norepeat, 'a', encoding='utf-8') as fio:
                fio.write(textbox)
                fio.write('\n')

        print(f"{file_prefix} saved to {save_path_norepeat}")


def print_df( df, title:str=None):
    if title is not None:
        print(title)
    print("\n"+tabulate(df, headers='keys', tablefmt='psql', floatfmt=(".4f")))