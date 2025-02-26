from src import AudioFileProcessor, DataProcessor, NoiseHandler, WhisperSTT, SpeakerDiarizer, TimeProcessor, ResultMapper
from src import LLMOpenAI
from dotenv import load_dotenv
import tempfile
import logging
import argparse
import time
import json
import os

logger = logging.getLogger('stt_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('stt-file-result.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def main(args):
    stt_file_name = "stt_" + args.file_name.split('.')[1].split('/')[-1] + '.json'
    output_file_name = "output_" + args.file_name.split('.')[1].split('/')[-1] + '.json'
    audio_file_path = args.file_name    
    
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
    hf_api_key = os.getenv('HF_API')

    audio_p = AudioFileProcessor()
    data_p = DataProcessor()
    time_p = TimeProcessor()
    diar_module = SpeakerDiarizer()
    diar_module.set_pyannotate(hf_api_key)
    resultMapper = ResultMapper()
    stt_module = WhisperSTT(openai_api_key)
    stt_module.set_client()
    stt_module.load_word_dictionary('./config/word_dict.json')

    start = time.time()
    diar_result = diar_module.seperate_speakers(audio_p, audio_file_path, num_speakers=4)
    results = [];  
    if args.chunk_length == None:
        result = stt_module.transcribe_text_api(audio_p, audio_file_path)
        with open(os.path.join('./data', stt_file_name), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"모든 결과가 JSON 파일 '{os.path.join('./data', stt_file_name)}'로 저장되었습니다.")
        print(f"소요 시간: {time.time() - start}")
    else:
        # filtered_audio = noise_handler.filter_audio_with_ffmpeg(audio_file_path, high_cutoff=50, low_cutoff=3500)
        audio_chunk = audio_p.audio_chunk(audio_file_path, chunk_length=args.chunk_length)
        for idx, chunk in enumerate(audio_chunk):
            chunk_offset = idx * args.chunk_length     # 청크의 시작 시간 오프셋 계산
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                chunk.export(temp_audio_file.name, format="wav")
                temp_audio_path = temp_audio_file.name

            stt_result = stt_module.transcribe_text_api(audio_p, temp_audio_path, logger=logger, chunk_offset=chunk_offset)
            results.append(stt_result)
        flatten_result = data_p.flatt_list(results)
        with open(os.path.join('./meeting_records', 'stt', stt_file_name), "w", encoding="utf-8") as output_file:
            json.dump(flatten_result, output_file, ensure_ascii=False, indent=4)
        
        for idx, stt_result in enumerate(flatten_result): 
            flatten_result[idx]['speaker_info'] = resultMapper.map_speaker_with_nested_check(time_p, stt_result, diar_result)
        
        with open(os.path.join('./meeting_records', output_file_name), "w", encoding="utf-8") as output_file:
            json.dump(flatten_result, output_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--output_path', type=str, default='./data/output')
    cli_parser.add_argument('--chunk_length', type=int, default=None)
    cli_parser.add_argument('--file_name', type=str, default=None) 
    cli_parser.add_argument('--participant', type=int, default=None)
    cli_args = cli_parser.parse_args()
    main(cli_args)