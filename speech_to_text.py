from src import NoiseHandler, VoiceEnhancer, VoiceSeperator, SpeakerDiarizer
from src import DataProcessor
from src import WhisperSTT
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
import argparse
import time
import os
import io 
import gc 


def main(args):
    file_name = "oneline-meeting.wav"
    audio_file_path = os.path.join(args.data_path, file_name)
    sep_file_path = os.path.join(args.output_path, 'vocal')
    chunk_file_path = os.path.join(args.data_path, 'chunk')
    
    speaker_info_pickle_path = os.path.join(args.output_path, 'speaker')
    save_path = os.path.join(args.data_path, 'stt')

    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
    hf_api_key = os.getenv('HF_API')

    openai_client = OpenAI(api_key=openai_api_key)
    data_p = DataProcessor()
    noise_handler = NoiseHandler()
    voice_enhancer = VoiceEnhancer()
    voice_seperator = VoiceSeperator()
    speaker_diarizer = SpeakerDiarizer(hf_api_key)
    stt_module = WhisperSTT(openai_client)
    stt_module.set_client()

    # voice_seperator.seperate_vocals_with_umix(audio_file_path, sep_file_path)   # 음성, 베이스, 건반 등 소리 분리  - 자원상 문제로 일단 스킵    
    start = time.time() 
    audio_chunk = data_p.audio_chunk(audio_file_path)
    processed_chunk = []
    for idx, chunk in enumerate(audio_chunk):
        speaker_info_pickle = f'sep-speaker-{idx}.pickle'
        nnnoise_chunk = noise_handler.remove_background_noise(chunk)
        print(f'노이즈 제거: {time.time() - start}')
        filtered_chunk = noise_handler.filter_audio_with_ffmpeg(nnnoise_chunk)
        nnnoise_chunk.close()
        print(f'오디오 주파수 필터링: {time.time() - start}')
        #emphasized_chunk = voice_enhancer.emphasize_nearby_voice(filtered_chunk)
        
        #print(f'근접 보이스 강조: {time.time() - start}', end='\n\n')
        speaker_diarizer.seperate_speakers(filtered_chunk, speaker_info_pickle_path, speaker_info_pickle, num_speakers=3)
        filtered_chunk.close()
        gc.collect()
        

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_args = cli_parser.parse_args()
    main(cli_args)