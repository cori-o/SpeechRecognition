'''
Time Alignment, Source Seperation, Speaker Diarization, Speaker Embedding
'''
from pydub.effects import high_pass_filter, low_pass_filter
from pyannote.audio import Pipeline
from pydub import AudioSegment
from io import BytesIO
import noisereduce as nr
import soundfile as sf
import numpy as np
import subprocess
import librosa
import tempfile
import pickle
import json
import os
import io 
import logging

logging.basicConfig(level=logging.WARNING)

class NoiseHandler: 
    '''
    음성 파일에서 노이즈를 제거한다.
    '''
    def remove_background_noise(self, input_file, output_file=None):
        """
        배경 잡음을 제거하여 가까운 목소리를 강조
        args:
            input_file (str): 입력 오디오 파일.
            output_file (str): 출력 오디오 파일.
        """
        try:
            y, sr = librosa.load(input_file, sr=None)   # 오디오 로드
        except:
            audio_buffer = io.BytesIO()
            input_file.export(audio_buffer, format="wav")
            audio_buffer.seek(0)  # 버퍼의 시작 위치로 이동
            y, sr = librosa.load(audio_buffer, sr=None)
        noise_profile = y[:sr]   # 배경 잡음 프로파일 추출 (초기 1초)
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile)   # 잡음 제거
        if output_file: 
            sf.write(output_file, y_denoised, sr)
            print(f"Saved denoised audio to {output_file}")
        else:
            wav_buffer = io.BytesIO()   # 메모리 내 WAV 파일 생성
            sf.write(wav_buffer, y_denoised, sr, format='WAV')
            wav_buffer.seek(0)   # 파일 포인터를 처음으로 이동
            return wav_buffer
    
    def filter_audio_with_pydub(self, input_file, high_cutoff=200, low_cutoff=3000, output_file=None):
        '''
        pydub (python)을 이용한 오디오 필터링 (고역대, 저역대)
        '''
        try:
            audio = AudioSegment.from_file(input_file)
        except:
            audio = input_file
        filtered_audio = high_pass_filter(audio, cutoff=high_cutoff)    # 고역 필터 (200Hz 이상)
        filtered_audio = low_pass_filter(filtered_audio, cutoff=low_cutoff)    # 저역 필터 (3000Hz 이하)
        if output_file:
            filtered_audio.export(output_file, format="wav")
            print(f"Filtered audio saved to {output_file}")
        else:
            audio_buffer = io.BytesIO()
            filtered_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
            return audio_buffer

    def filter_audio_with_ffmpeg(self, input_file, high_cutoff=200, low_cutoff=3000, output_file=None):
        """
        FFmpeg을 사용한 오디오 필터링 (고역대, 저역대).
        Args:
            input_file (str or BytesIO): 입력 오디오 파일 경로 또는 BytesIO 객체.
            high_cutoff (int): 고역 필터 컷오프 주파수 (Hz).
            low_cutoff (int): 저역 필터 컷오프 주파수 (Hz).
            output_file (str, optional): 필터링된 오디오 저장 경로. 지정되지 않으면 메모리로 반환.
        Returns:
            io.BytesIO: 필터링된 오디오 데이터 (output_file이 None인 경우).
        """
        if isinstance(input_file, io.BytesIO):  
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                temp_input.write(input_file.getvalue())
                temp_input.flush()
                input_source = temp_input.name
        elif isinstance(input_file, (str, os.PathLike)):
            input_source = input_file
        try:
            command = [
                "ffmpeg",
                "-i", input_source,  # 입력 파일
                "-af", f"highpass=f={high_cutoff},lowpass=f={low_cutoff}",  # 필터 적용
                "-f", "wav",  # 출력 형식
                "pipe:1" if not output_file else output_file  # 메모리로 반환하거나 파일로 저장
            ]
            if output_file:
                subprocess.run(command, check=True)
                print(f"Filtered audio saved to {output_file}")
            else:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
                return io.BytesIO(stdout)  # BytesIO로 반환
        finally:
            # 임시 파일 삭제
            if isinstance(input_file, io.BytesIO) and 'temp_input' in locals():
                os.unlink(temp_input.name)


class VoiceEnhancer:
    '''
    음성 파일에서 음성을 강화한다. 
    '''
    def emphasize_nearby_voice(self, input_file, threshold=0.05, output_file=None):
        """
        가까운 음성을 강조하고 먼 목소리를 줄임
        args:
            input_file (str): 입력 오디오 파일
            output_file (str): 출력 오디오 파일
            threshold (float): 에너지 기준값 (낮을수록 약한 신호 제거)
        """
        try:
            y, sr = librosa.load(input_file, sr=None)   # 오디오 로드
        except:
            audio_buffer = io.BytesIO()
            input_file.export(audio_buffer, format="wav")
            audio_buffer.seek(0)  # 버퍼의 시작 위치로 이동
            y, sr = librosa.load(audio_buffer, sr=None)           
        rms = librosa.feature.rms(y=y)[0]         # RMS 에너지 계산
        mask = rms > threshold                    # 에너지 기준으로 마스크 생성

        expanded_mask = np.repeat(mask, len(y) // len(mask) + 1)[:len(y)]   # RMS 값을 전체 신호 길이에 맞게 확장
        y_filtered = y * expanded_mask.astype(float)   # 입력 신호에 확장된 마스크 적용

        if output_file:
            sf.write(output_file, y_filtered, sr)   # 강조된 오디오 저장
            print(f"Saved emphasized audio to {output_file}")
        else:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, y_filtered, sr, format="WAV")
            audio_buffer.seek(0)  # 버퍼의 시작 위치로 이동
            return audio_buffer


class VoiceSeperator:
    '''
    음성 파일에서 vocal, drum, base 등의 소리 분리
    '''
    def separate_vocals_with_demucs(self, input_file, output_dir):
        subprocess.run([
            "demucs",
            "--two-stems", "vocals",  # 보컬만 분리
            "--out", output_dir,     # 출력 경로
            input_file       # 입력 파일
        ], check=True)
        print(f"Separated vocals saved in {output_dir}")

    def seperate_vocals_with_umix(self, input_file, output_file):
        subprocess.run([
            "umx",
            "--outdir", output_file,
            input_file
        ], check=True)
        print(f"Seperated vocals save in {output_file}")


class SpeakerDiarizer:
    def __init__(self, hf_api):
        self.hf_api = hf_api
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=self.hf_api)

    def bytesio_to_tempfile(self, audio_bytesio):
        """
        BytesIO 객체를 임시 파일로 저장.
        args:
            audio_bytesio (BytesIO): BytesIO 객체.
        returns:
            str: 임시 파일 경로.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytesio.getvalue())
            temp_file.flush()
            return temp_file.name

    def convert_segments(self, result):
        """
        화자 분리 결과를 적절한 형식으로 변환.
        args:
            result (tuple): Pyannote diarization 결과.
        returns:
            dict: 변환된 결과.
        """
        segment, _, speaker = result
        return {
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker
        }

    def seperate_speakers(self, audio_file, save_path, file_name, num_speakers=None):
        """
        화자 분리 실행 및 결과 저장.
        args:
            audio_file (str or BytesIO): 입력 오디오 파일 경로 또는 BytesIO 객체.
            save_path (str): 결과 저장 경로.
            file_name (str): 저장할 파일 이름.
            num_speakers (int, optional): 예상 화자 수. 기본값은 Pyannote에서 자동 감지.
        """
        print(isinstance(audio_file, io.BytesIO))
        if isinstance(audio_file, io.BytesIO):   # 입력 데이터 형식 확인 및 변환
            audio_file = self.bytesio_to_tempfile(audio_file)

        try:   # Pyannote Pipeline 초기화
            diarization = self.pipeline(audio_file, num_speakers=num_speakers)
        except Exception as e:
            print(f"[ERROR] Diarization failed: {e}")
            return

        # 결과 변환 및 저장
        results = []
        for result in diarization.itertracks(yield_label=True):  # result: (<Segment>, _, speaker)
            converted_info = self.convert_segments(result)
            results.append(converted_info)

        # 저장 경로 확인 및 결과 저장
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, file_name)
        with open(save_file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_file_path}")