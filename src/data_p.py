'''
audio chunking, pcm convert
'''
from pydub import AudioSegment
import soundfile as sf
import numpy as np 
import librosa
import pickle
import wave
import os


class DataProcessor:
    def data_to_df(self, dataset, columns):
        if isinstance(dataset, list):
            return pd.DataFrame(dataset, columns=columns)
       
    def df_to_hfdata(self, df):
        return Dataset.from_pandas(df)

    def merge_data(self, df1, df2, how='inner', on=None):
        return pd.merge(df1, df2, how='inner')

    def filter_data(self, df, col, val):
        return df[df[col]==val].reset_index(drop=True)
    
    def remove_keywords(self, df, col, keyword=None, exceptions=None):
        if exceptions != None: 
            if keyword != None:
                # pattern = r'(?<![\w가-힣])(?:' + '|'.join(map(re.escape, keyword)) + r')(?![\w가-힣])'
                pattern = re.compile(r'(?<![\w가-힣])(' + '|'.join(map(re.escape, val)) + r')(?=[^가-힣]|$)')
            else:
                pattern = r'(?<![\w가-힣])(\S*주)(?![\w가-힣])'    # 테마주 같은 함정 증권 종목 제거 
            mask = df[col].str.contains(pattern, na=False) & ~df[col].str.contains('|'.join(map(re.escape, exceptions)), na=False)
            df = df[~mask]
            return df.reset_index(drop=True)
        else: 
            keyword_idx = df[df[col].str.contains(keyword, na=False)].index 
            df.drop(keyword_idx, inplace=True)
            return df.reset_index(drop=True)
        
    def train_test_split(self, dataset, x_col, y_col, test_size, val_test_size, random_state=42):
        X, X_test, y, y_test = train_test_split(dataset[x_col], dataset[y_col], test_size=0.2, stratify=dataset[y_col], random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_test_size, stratify=y, random_state=random_state)
        return X, X_val, X_test, y, y_val, y_test
        
    def save_results_to_pickle(self, result, output_file):
        with open(output_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Results saved to {output_file}")

    def load_results_from_pickle(self, input_file):
        with open(input_file, "rb") as f:
            result = pickle.load(f)
        print(f"Results loaded from {input_file}")
        return result