from src import NoiseHandler, VoiceEnhancer, VoiceSeperator, SpeakerDiarizer
from src import DataProcessor, AudioFileProcessor
from src import WhisperSTT
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
import argparse
import time
import json
import os
import gc

def main(args):
    file_name = args.file_name
    audio_file_path = os.path.join(args.data_path, file_name)    

    audio_p = AudioFileProcessor()
    audio_chunk = audio_p.audio_chunk(audio_file_path, chunk_length=270, chunk_file_path=os.path.join(args.output_path, 'chunk'), chunk_file_name='chunk_' + args.file_name.split('.')[0])


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--file_name', type=str, default='oneline-meeting.wav')
    cli_args = cli_parser.parse_args()
    main(cli_args)