import io 
from pydub import AudioSegment
from openai import OpenAI
from abc import ABC, abstractmethod
import tempfile

class STTModule:
    def __init__(self, openai_api=None, ms_api=None):
        self.openai_api = openai_api
        self.ms_api = ms_api 
    
    @abstractmethod
    def set_client(self):
        pass

    @abstractmethod
    def convert_text_to_speech(self, audio_path, save_path):
        pass 


class WhisperSTT(STTModule):
    def __init__(self, openai_api):
        super().__init__(openai_api=openai_api)
    
    def set_client(self):
        self.openai_client = OpenAI(api_key=self.openai_api)

    def process_segments_with_whisper(self, audio_file, segments):
        """
        화자 구간을 나눠 Whisper API에 바로 전달하여 텍스트 변환
        args:
            audio_file (str): 입력 오디오 파일 경로
            segments (List[Tuple[float, float, str]]): (시작 시간, 종료 시간, 화자) 세그먼트
        """
        audio = AudioSegment.from_file(audio_file)
        for i, (start, end, speaker) in enumerate(segments):
            segment_duration = end - start
            if segment_duration < 0.1:
                print(f"Skipping segment {i}: Duration too short ({segment_duration:.2f}s)")
                continue

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
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
                            file=audio_file
                        )
                    print(f"Speaker {speaker}, Segment {i}: {transcription.text}")
            except Exception as e:
                print(f"Error processing segment {i} for Speaker {speaker}: {e}")