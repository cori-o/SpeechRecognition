from src import AudioFileProcessor
import argparse
import os

def main(args):
    audio_fp = AudioFileProcessor()
    file_path = os.path.join(args.data_path, args.file_name)
    audio_fp.m4a_to_wav(file_path)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--file_name', type=str, default=None)
    cli_args = cli_parser.parse_args()
    main(cli_args)