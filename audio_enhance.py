from src import AudioFileProcessor, NoiseHandler, VoiceEnhancer, VoiceSeperator, SpeakerDiarizer
from src import DataProcessor
from src import WhisperSTT
from dotenv import load_dotenv
from openai import OpenAI
import argparse
import pickle
import os

def main(args):
    file_name = args.file_name
    audio_file_path = os.path.join(args.data_path, args.file_name)

    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
    hf_api_key = os.getenv('HF_API')
    audio_p = AudioFileProcessor()
    noise_handler = NoiseHandler()
    voice_enhancer = VoiceEnhancer()
    filtered_audio = noise_handler.filter_audio_with_ffmpeg(audio_file_path, high_cutoff=150, low_cutoff=5000)
    # print(filtered_audio)
    nnnoise_audio = noise_handler.remove_background_noise(filtered_audio, output_file=os.path.join(args.data_path, 'lufs_denoised_' + args.file_name), prop_decrease=0.7)
    # nnnoise_audio = noise_handler.remove_noise_with_ffmpeg(filtered_audio, output_file=os.path.join(args.data_path, 'lufs_ffmpeg_denoised_' + args.file_name))
    normalized_audio = voice_enhancer.normalize_audio_lufs(nnnoise_audio, output_file=os.path.join(args.data_path, 'lufs_norm_' + args.file_name))
    
    # print(nnnoise_audio)
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--file_name', type=str, default=None)

    cli_args = cli_parser.parse_args()
    main(cli_args)