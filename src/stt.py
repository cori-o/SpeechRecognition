from pydub import AudioSegment
from openai import OpenAI
from abc import ABC, abstractmethod
import tempfile
import json
import io
import re

class STTModule:
    def __init__(self, openai_api_key=None, ms_api=None):
        self.openai_api_key = openai_api_key
        self.ms_api = ms_api 
    
    @abstractmethod
    def set_client(self):
        pass

    @abstractmethod
    def convert_text_to_speech(self, audio_path, save_path):
        pass 


class WhisperSTT(STTModule):
    def __init__(self, openai_api_key):
        super().__init__(openai_api_key=openai_api_key)
    
    def set_client(self):
        self.openai_client = OpenAI(api_key=self.openai_api_key)

    def load_word_dictionary(self, json_path):
        with open(json_path, mode='r', encoding='utf-8') as file:
            self.word_dict = json.load(file)
            return json.load(file)
    
    def apply_word_dictionary(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = stt_text.replace(incorrect_word, correct_word)
        return stt_text
    
    def apply_word_dictionary_regex(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = re.sub(rf'\b{re.escape(incorrect_word)}\b', correct_word, stt_text)
        return stt_text
    
    def process_whisper(self, data_p, audio_file):
        '''
        
        '''

    def process_segments_with_whisper(self, data_p, audio_file, segments):
        """
        화자 구간을 나눠 Whisper API에 바로 전달하여 텍스트 변환
        args:
            audio_file (str): 입력 오디오 파일 경로
            segments (List[Tuple[float, float, str]]): (시작 시간, 종료 시간, 화자) 세그먼트
        """
        if isinstance(audio_file, io.BytesIO):   # 입력 데이터 형식 확인 및 변환
            audio_file = data_p.bytesio_to_tempfile(audio_file)
        
        results = []
        exclude_word = ['뉴스', '구독', '좋아요']
        audio = AudioSegment.from_file(audio_file)
        for i, segment in enumerate(segments):
            # segment_duration = segment['end'] - segment['start']
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            segment_audio = audio[start_ms:end_ms]
            segment_audio = segment_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            audio_buffer = io.BytesIO()
            segment_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)    # 파일 포인터를 처음으로 이동
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    segment_audio.export(temp_audio_file.name, format="wav")
                    with open(temp_audio_file.name, "rb") as audio_file:
                        transcription = self.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language='ko',
                            # prompt="이 대화는 이뤄지는 한국어 대화입니다."
                        )
                    modified_text = self.apply_word_dictionary(transcription.text, self.word_dict)   
                    pattern = r'(^|\s|[^가-힣a-zA-Z0-9])(' + '|'.join(map(re.escape, exclude_word)) + r')($|\s|[^가-힣a-zA-Z0-9])'
                    if re.search(pattern, modified_text):
                        continue
                    results.append({
                        "speaker": segment["speaker"],
                        "start_time": round(segment["start"], 2),
                        "end_time": round(segment["end"], 2),
                        "text": modified_text
                    })
            except Exception as e:
                print(f"Error processing segment {i} for Speaker {segment['speaker']}: {e}")
        return results