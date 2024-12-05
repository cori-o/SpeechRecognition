from dotenv import load_dotenv
from src import AudioProcessor, DataProcessor
from pydub import AudioSegment
import argparse
import os 

def main(args):
    # audio_file_path = os.path.join(args.data_path, 'oneline-meeting.wav')
    # audio = AudioSegment.from_file(audio_file_path)

    data_p = DataProcessor()
    audio_p = AudioProcessor()
    
    # data_p.audio_chunk(audio_file_path, os.path.join(args.output_path, 'chunk'), 'oneline-mt-chunk')
    audio_file_path = os.path.join(args.data_path, 'chunk', 'oneline-mt-chunk_0.wav')
    # audio_p.seperate_vocals_with_umix(audio_file_path, os.path.join(args.output_path, 'vocal'))
    vocal_file_path = os.path.join(args.output_path, 'vocal', 'oneline-mt-chunk_0', 'vocals.wav')
    audio_p.seperate_speakers(vocal_file_path)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_args = cli_parser.parse_args()
    main(cli_args)