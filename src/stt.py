from pydub.silence import detect_nonsilent
from abc import ABC, abstractmethod
from pydub import AudioSegment
from openai import OpenAI
import tempfile
import json
import io
import os
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
            self.word_dict = json.load(file)  # JSON 데이터를 한번만 로드
            return self.word_dict
    
    def apply_word_dictionary(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = stt_text.replace(incorrect_word, correct_word)
        return stt_text
    
    def apply_word_dictionary_regex(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = re.sub(rf'\b{re.escape(incorrect_word)}\b', correct_word, stt_text)
        return stt_text
    
    def filter_stt_result(self, results):
        filtered_results = []
        prev_text = None
        for segment in results:
            text = segment['text'].strip()
            if text == prev_text:  # 이전 텍스트와 동일하면 제거
                continue
            prev_text = text
            filtered_results.append(segment)
        return filtered_results

    def calculate_nonsilent_start(self, audio_file_path, silence_thresh=-30, min_silence_len=500):
        """
        Calculate the start time of the first non-silent portion in an audio file.
        args:
            audio_file_path (str): Path to the audio file.
            silence_thresh (int): Silence threshold in dBFS.
            min_silence_len (int): Minimum silence length in ms.   
        returns:
            float: Start time of the first non-silent portion in seconds.
        """
        audio = AudioSegment.from_file(audio_file_path)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        start_time_ms = nonsilent_ranges[0][0] if nonsilent_ranges else 0
        return start_time_ms / 1000  # Convert ms to seconds
    
    def map_speaker_with_nested_check(self, time_p, diar_segments, stt_result):
        """STT 구간과 Diarization 구간 비교 후, 중첩된 구간 확인"""
        stt_start, stt_end = stt_result['start_time'], stt_result['end_time']
        stt_duration = stt_end - stt_start
        candidates = []

        for diar_seg in diar_segments:   # STT 결과값과 겹치는 Diar 구간 탐색 
            diar_start, diar_end = diar_seg['start'], diar_seg['end']
            if stt_start <= diar_end and stt_end >= diar_start:  # 겹침 조건
                candidates.append(diar_seg)                
        print(f'candidate: {candidates}')

        for candidate in candidates:
            diar_start, diar_end = candidate['start'], candidate['end']
            nested_segments = [  #  Diar 구간 내에 또 다른 구간 탐색 (0~19 -> 13~14)
                seg for seg in diar_segments
                if seg['start'] >= diar_start and seg['end'] <= diar_end and seg != candidate
            ]
            if nested_segments:   # 또 다른 발화가 있을 때 
                for nested in nested_segments:
                    if time_p.is_similar(nested, stt_segment):
                        stt_result['speaker'] = nested['speaker']
                        return stt_result
            max_overlap = 0
            best_speaker = 'Unknown'
            for diar_seg in candidates:
                overlap_start = max(stt_start, diar_seg['start'])
                overlap_end = min(stt_end, diar_seg['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = diar_seg['speaker']
                    print(f'best_speaker: {best_speaker}')
            stt_result['speaker'] = best_speaker
            return stt_result

    def process_segments_with_whisper(self, audio_p, audio_file, diar_results, db_stt_result_path, meeting_id, table_editor, openai_client):
        """
        화자 구간을 나눠 Whisper API에 바로 전달하여 텍스트 변환
        args:
            audio_file (str): 입력 오디오 파일 경로
            segments (List[Tuple[float, float, str]]): (시작 시간, 종료 시간, 화자) 세그먼트
        """
        if isinstance(audio_file, AudioSegment):
            whisper_audio = audio_file.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_buffer = io.BytesIO()
            whisper_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
            audio_file = audio_buffer
        elif isinstance(audio_file, io.BytesIO):
            audio_file = audio_p.bytesio_to_tempfile(audio_file)
        
        exclude_word = ['뉴스', '구독', '시청']
        pattern = r'(^|\s|[^가-힣a-zA-Z0-9])(' + '|'.join(map(re.escape, exclude_word)) + r')($|\s|[^가-힣a-zA-Z0-9])'
        now = datetime.now()
        results = []
        audio = AudioSegment.from_file(audio_file)
        for idx, diar_result in enumerate(diar_results):
            diar_duration = diar_result['end'] - diar_result['start']
            start_time = now + timedelta(seconds=diar_result['start'])
            end_time = now + timedelta(seconds=diar_result['end'])
            if diar_duration < 0.1:
                print(f"Skipping segment {idx}: Duration too short ({diar_duration:.2f}s)")
                continue
            if diar_duration < 2:
                try:
                    start_ms = int(diar_result['start'] * 1000) - 1
                    end_ms = int(diar_result['end'] * 1000) + 1
                except:
                    start_ms = int(diar_result['start'] * 1000)
                    end_ms = int(diar_result['end'] * 1000)
            else:
                start_ms = int(diar_result['start'] * 1000)
                end_ms = int(diar_result['end'] * 1000)
            segment_audio = audio[start_ms:end_ms]


    def transcribe_text(self, audio_p, audio_file):
        if isinstance(audio_file, AudioSegment):
            whisper_audio = audio_file.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_buffer = io.BytesIO()
            whisper_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
            audio_file = audio_buffer
        elif isinstance(audio_file, io.BytesIO):
            audio_file = audio_p.bytesio_to_tempfile(audio_file)
        
        audio = AudioSegment.from_file(audio_file)  # 컨텍스트 매니저 제거
        whisper_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        self.load_word_dictionary(os.path.join('./config', 'word_dict.json'))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            whisper_audio.export(temp_audio_file.name, format="wav")
            with open(temp_audio_file.name, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language='ko',
                    response_format="verbose_json",
                    #timestamp_granularities=["segment"]
                )
        segments = transcription.segments
        print(f'trans result: {transcription.segments}')   # id, avg_logprob, compression_ratio, end, no_speech_prob, seek, start, temperature (0.0), text, tokens
        results = []
        for segment in segments:
            if segment.no_speech_prob < 0.9 and segment.avg_logprob > -3.3:
                modified_text = self.apply_word_dictionary(segment.text, self.word_dict)
                results.append({
                    'start_time': round(segment.start, 5),   # + nonsilent_s  제거. 초반만 이상하고, 후반에는 원상 복귀됨 
                    'end_time': round(segment.end, 5),   # + nonsilent_s 
                    'text': modified_text.strip(),
                    'prob': segment.no_speech_prob,
                    'avg_logprob': segment.avg_logprob
                })
        filtered_results = self.filter_stt_result(results)
        return filtered_results