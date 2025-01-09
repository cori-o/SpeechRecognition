import pandas as pd
import json
import os

data = pd.read_csv(os.path.join('./config/word_dict.csv'), header=None, names=['원래 단어', '수정 단어'])
sorted_df = data.sort_values(by="수정 단어").reset_index(drop=True)   # "수정 단어" 기준으로 정렬
word_dict = sorted_df.set_index('원래 단어')['수정 단어'].to_dict()   # DataFrame을 딕셔너리로 변환

output_path = './config/word_dict.json'
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(word_dict, file, ensure_ascii=False, indent=4)