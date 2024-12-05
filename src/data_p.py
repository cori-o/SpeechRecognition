'''
audio chunking, pcm convert
'''
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter
import subprocess
import wave
import os


class DataProcessor:
    def audio_chunk(self, audio_file_path, output_path, chunk_file_name, chunk_length=60):
        audio = AudioSegment.from_file(audio_file_path)
        chunk_length_ms = chunk_length * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        for idx, chunk in enumerate(chunks):
            temp_file_path = os.path.join(output_path, f"{chunk_file_name}_{idx}.wav")
            chunk.export(temp_file_path, format="wav")
    
    def pcm_to_wav(self, pcm_file_path, wav_file_path, sample_rate=44100, channels=1, bit_depth=16):
        try:
            with open(pcm_file_path, 'rb') as pcm_file:   # PCM 파일 열기
                pcm_data = pcm_file.read()
            
            # WAV 파일 생성
            with wave.open(wav_file_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)  # 채널 수 (1: 모노, 2: 스테레오)
                wav_file.setsampwidth(bit_depth // 8)  # 샘플 당 바이트 수
                wav_file.setframerate(sample_rate)  # 샘플링 속도
                wav_file.writeframes(pcm_data)  # PCM 데이터 쓰기

            print(f"WAV 파일이 성공적으로 생성되었습니다: {wav_file_path}")
        except Exception as e:
            print(f"오류 발생: {e}")

    
    def filter_audio_with_pydub(self, input_file, output_file, high_cutoff=200, low_cutoff=3000):
        audio = AudioSegment.from_file(input_file)

        filtered_audio = high_pass_filter(audio, cutoff=high_cutoff)    # 고역 필터 (200Hz 이상)
        filtered_audio = low_pass_filter(filtered_audio, cutoff=low_cutoff)    # 저역 필터 (3000Hz 이하)
        filtered_audio.export(output_file, format="wav")
        print(f"Filtered audio saved to {output_file}")

    def apply_audio_filters(self, input_file, output_file):
        '''
        ffmpeg (C++)을 이용한 오디오 필터링 (고역대, 저역대)
        '''
        try:
            subprocess.run([   # ffmpeg 명령어 실행
                "ffmpeg",
                "-i", input_file,  # 입력 파일
                "-af", "highpass=f=200, lowpass=f=3000",  # 오디오 필터 적용
                output_file
            ], check=True)
            print(f"Filtered audio saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error during ffmpeg processing: {e}")