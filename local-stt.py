from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src import WhisperSTT
from datasets import load_dataset
from dotenv import load_dotenv
import tempfile
import logging
import argparse
import torch
import time
import json
import os

def main(args):
    stt_file_name = "stt_update_" + args.file_name.split('.')[1].split('/')[-1] + '.json'
    audio_file_path = args.file_name    
    
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')

    stt_module = WhisperSTT(openai_api_key)
    stt_module.set_client()
    stt_module.set_gpu_device()
    stt_module.set_generate_kwargs()
    stt_module.load_word_dictionary('./config/word_dict.json')
    model_id = "openai/whisper-large-v3"
    result = stt_module.transcribe_text(model_id, audio_file_path)
    # print(result)
    with open(os.path.join('./meeting_records', 'stt', stt_file_name), "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--chunk_length', type=int, default=None)
    cli_parser.add_argument('--file_name', type=str, default=None) 
    cli_parser.add_argument('--participant', type=int, default=None)
    cli_args = cli_parser.parse_args()
    main(cli_args)