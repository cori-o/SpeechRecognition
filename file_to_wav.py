from src import AudioFileProcessor
import argparse
import os

def main(args):
    audio_fp = AudioFileProcessor()
    wav_file_name = args.file_name.split('.')[0] + '.wav'
    file_path = os.path.join(args.data_path, args.file_name)
    audio_fp.m4a_to_wav(file_path)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--file_name', type=str, default=None)
    cli_parser.add_argument('--file_type', type=str, default='pcm')
    cli_args = cli_parser.parse_args()
    main(cli_args)