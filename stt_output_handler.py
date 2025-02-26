from src import DataProcessor
from dotenv import load_dotenv
import argparse
import json
import os
import pandas as pd


def main(args):
    stt_file_name = "stt_update_" + args.file_name.split('.')[1].split('/')[-1] + '.json'
    
    load_dotenv()
    data_p = DataProcessor()
    with open(os.path.join('./meeting_records', 'stt', stt_file_name), "r", encoding="utf-8") as f:
        stt_result = json.load(f)
    
    last_time = 0; 
    prev_text = ''; prev_s = 0; prev_e = 0
    new_timestamp = []; new_text = []
    for _ , chunk in enumerate(stt_result['chunks']):
        origin_s = chunk['timestamp'][0]; origin_e = chunk['timestamp'][-1]
        chunk['timestamp'][0] = chunk['timestamp'][0] + last_time
        if chunk['text'] == '':    
            chunk['timestamp'][-1] = chunk['timestamp'][0]
            last_time = chunk['timestamp'][0]      # ['timestamp'][-1] == 0인 경우가 대다수. ['timestamp'][0] = 28.0 등 
            prev_s = origin_s; prev_e = origin_e      # 30, 0 
            continue
        elif prev_e > origin_s:    # 13 > 1 (새로운 청크로 넘어감)
            last_time = last_time + prev_e
            chunk['timestamp'][0] = last_time + origin_s 
            chunk['timestamp'][-1] = last_time + origin_e
            last_time = chunk['timestamp'][-1]
        elif chunk['text'] != '' and chunk['timestamp'][-1] == 0:   # end time이 0인데, 실제 대화가 들어있는 경우
            chunk['timestamp'][-1] = chunk['timestamp'][0] + (30 - origin_e)
            last_time = chunk['timestamp'][-1]
        else:    # 일반적인 대화 상황 
            try:
                chunk['timestamp'][-1] = chunk['timestamp'][-1] + last_time
            except:    # end_time = Null
                chunk['timestamp'][-1] = chunk['timestamp'][0]           
        rr_text = data_p.remove_repeated_patterns(chunk['text'].strip())
        cleansed_text = data_p.cleanse_text(rr_text)
        if prev_text == cleansed_text or cleansed_text.isdigit(): 
            continue
        prev_text = cleansed_text 
        new_timestamp.append((chunk['timestamp'][0], chunk['timestamp'][-1]))
        new_text.append(cleansed_text)
        prev_s = origin_s; prev_e = origin_e
    pd.DataFrame(zip(new_timestamp, new_text), columns=['timestamp', 'text']).to_csv('./meeting_records/test.csv', index=False)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--chunk_length', type=int, default=None)
    cli_parser.add_argument('--file_name', type=str, default=None) 
    cli_parser.add_argument('--participant', type=int, default=None)
    cli_args = cli_parser.parse_args()
    main(cli_args)