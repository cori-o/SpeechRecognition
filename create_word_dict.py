import pandas as pd
import json
import os

data = pd.read_csv(os.path.join('./data/word_dict.csv'), header=None, names=['원래 단어', '수정 단어'])
sorted_df = data.sort_values(by="수정 단어").reset_index(drop=True)
print(sorted_df)