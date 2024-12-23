from src import NoiseHandler, VoiceEnhancer, VoiceSeperator, SpeakerDiarizer
from src import DataProcessor
from src import WhisperSTT, MSSTT
from dotenv import load_dotenv
from openai import OpenAI
import argparse
import pickle
import os


def main(args):
    file_name = "oneline-meeting.wav"
    audio_file_path = os.path.join(args.data_path, file_name)
    chunk_file_name = file_name.split('.')[0] + "_chunk"
    chunk_file_path = os.path.join(args.data_path, 'chunk')
    sep_file_path = os.path.join(args.data_path, 'sep')
    vocal_file_path = os.path.join(args.output_path, 'vocal', f'oneline-mt-chunk_{file_no}', 'vocals.wav')
    speaker_info_pickle_path = os.path.join(args.output_path, 'speaker')
    speaker_info_pickle = f'sep-speaker-{file_no}.pickle'
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

    file_no = 0
    noise_handler.remove_background_noise(audio_file_path, os.path.jo)
    noise_handler.filter_audio_with_pydub(audio_file_path, '')
    voice_enhancer.emphasize_nearby_voice(audio_file_path, )
    data_p.audio_chunk(audio_file_path, chunk_file_path, chunk_file_name)    # file 자르기  - whisper stt 사용하기 위함 
    voice_seperator.seperate_vocals_with_umix(audio_file_path, sep_file_path)    # 음성, 베이스, 건반 등 소리 분리 
    speaker_diarizer.seperate_speakers(vocal_file_path, speaker_info_pickle_path, speaker_info_pickle, num_speakers=3)   
    with open(os.path.join(speaker_info_pickle_path, speaker_info_pickle), "rb") as f:
        result = pickle.load(f)
    stt_module.process_segments_with_whisper(vocal_file_path, result)

    
if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--data_path', type=str, default='./data')
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_args = cli_parser.parse_args()
    main(cli_args)