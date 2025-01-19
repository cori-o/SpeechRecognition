from src import SpeakerDiarizer
from dotenv import load_dotenv
import argparse
import json
import os

def main(args):
    load_dotenv()
    hf_api_key = os.getenv('HF_API')

    speaker_diarizer = SpeakerDiarizer()
    speaker_diarizer.set_pyannotate(hf_api_key)
    with open(os.path.join('./data', args.file_name), "r", encoding="utf-8") as f:
        diar_result = json.load(f)
    print(speaker_diarizer.calc_speak_duration(diar_result, 'SPEAKER_02'))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--chunk_length', type=int, default=None)
    cli_parser.add_argument('--file_name', type=str, default=None) 
    cli_args = cli_parser.parse_args()
    main(cli_args)