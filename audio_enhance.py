from src import NoiseHandler, VoiceEnhancer
from dotenv import load_dotenv
import argparse
import os


def main(args):
    audio_file_path = os.path.join(args.data_path, args.file_name)

    noise_handler = NoiseHandler()
    voice_enhancer = VoiceEnhancer()
    filtered_audio = noise_handler.filter_audio_with_ffmpeg(audio_file_path, high_cutoff=100, low_cutoff=3500)
    nnnoise_audio = noise_handler.remove_background_noise(filtered_audio, output_file=os.path.join(args.data_path, 'lufs_denoised_' + args.file_name), prop_decrease=0.8)
    # normalized_audio = voice_enhancer.normalize_audio_lufs(nnnoise_audio, output_file=os.path.join(args.data_path, 'lufs_norm_' + args.file_name))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--file_name', type=str, default=None)
    cli_args = cli_parser.parse_args()
    main(cli_args)