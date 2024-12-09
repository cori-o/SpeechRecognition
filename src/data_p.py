'''
audio chunking, pcm convert
'''
from pydub import AudioSegment
import soundfile as sf
import numpy as np 
import librosa
import pickle
import wave
import os


class DataProcessor:
    def align_audio(self, reference_file, target_file, output_file):
        lag, sr = self.calculate_time_lag(reference_file, target_file)   # 시간차 계산
        target, _ = librosa.load(target_file, sr=sr)   # 오디오 로드
        aligned_target = np.pad(target, (lag, 0), mode='constant') if lag > 0 else target[-lag:]
        sf.write(output_file, aligned_target, sr)     # 정렬된 오디오 저장
        print(f"Aligned audio saved to {output_file}")

    def calculate_time_lag(self, reference_file, target_file):
        ref, sr_ref = librosa.load(reference_file, sr=None)     # 샘플링 속도 유지, 기준 오디오 
        target, sr_target = librosa.load(target_file, sr=None)    # 타겟 오디오 로드

        if sr_ref != sr_target:   # 샘플링 속도 확인 및 일치
            raise ValueError("Sample rates of the audio files must be the same.")
        correlation = np.correlate(target, ref, mode='full')   # Cross-Correlation 계산
        lag = np.argmax(correlation) - len(ref)
        return lag, sr_ref

    def audio_chunk(self, audio_file_path, chunk_length=60, chunk_file_path=None, chunk_file_name=None):
        audio = AudioSegment.from_file(audio_file_path)
        chunk_length_ms = chunk_length * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        if chunk_file_path:
            for idx, chunk in enumerate(chunks):
                temp_file_path = os.path.join(chunk_file_path, f"{chunk_file_name}_{idx}.wav")
                chunk.export(temp_file_path, format="wav")
        else:
            return chunks
    
    def concat_chunk(self, chunk_list, save_path=None):
        final_audio = sum(chunk_list)
        if save_path:    
            final_audio.export("processed_audio.wav", format="wav")
        else:
            return final_audio
    
    def pcm_to_wav(self, pcm_file_path, wav_file_path, sample_rate=44100, channels=1, bit_depth=16):
        try:
            with open(pcm_file_path, 'rb') as pcm_file:   # PCM 파일 열기
                pcm_data = pcm_file.read()
            
            with wave.open(wav_file_path, 'wb') as wav_file:   # WAV 파일 생성
                wav_file.setnchannels(channels)   # 채널 수 (1: 모노, 2: 스테레오)
                wav_file.setsampwidth(bit_depth // 8)   # 샘플 당 바이트 수
                wav_file.setframerate(sample_rate)   # 샘플링 속도
                wav_file.writeframes(pcm_data)   # PCM 데이터 쓰기
            print(f"WAV 파일이 성공적으로 생성되었습니다: {wav_file_path}")
        except Exception as e:
            print(f"오류 발생: {e}")

    '''
    def split_audio_by_segments(self, audio_file: str, segments: List[Segment], output_dir: str):
        """ 
        args:
            audio_file (str): 원본 오디오 파일 경로.
            segments (List[Segment]): (시작 시간, 종료 시간, 화자) 리스트.
            output_dir (str): 분할된 파일 저장 경로.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio = AudioSegment.from_file(audio_file)
        
        for i, (start, end, speaker) in enumerate(segments):
            start_ms = int(start * 1000)    # ms 단위로 변환
            end_ms = int(end * 1000)
            segment_audio = audio[start_ms:end_ms]

            segment_file = os.path.join(output_dir, f"{speaker}_{i}.wav")
            segment_audio.export(segment_file, format="wav")
            print(f"Saved {segment_file}")'''

    def convert_segments(self, segment_data):
        """
        Segment 데이터를 변환.
        Args:
            segment_data (Tuple[Segment, str, str]): Pyannote diarization 결과의 단일 항목.
        Returns:
            Tuple[float, float, str]: 변환된 (start, end, speaker) 데이터.
        """
        segment, _, speaker = segment_data   # 튜플 unpack
        return (float(segment.start), float(segment.end), speaker)
        
    def save_results_to_pickle(self, result, output_file):
        with open(output_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Results saved to {output_file}")

    def load_results_from_pickle(self, input_file):
        with open(input_file, "rb") as f:
            result = pickle.load(f)
        print(f"Results loaded from {input_file}")
        return result