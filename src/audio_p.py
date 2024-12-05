'''
Time Alignment   
Source Seperation
Speaker Diarization
Speaker Embedding
'''
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np
import subprocess
import librosa
import torch

class AudioProcessor:
    def calculate_time_lag(self, reference_file, target_file):
        ref, sr_ref = librosa.load(reference_file, sr=None)     # 샘플링 속도 유지, 기준 오디오 
        target, sr_target = librosa.load(target_file, sr=None)    # 타겟 오디오 로드

        if sr_ref != sr_target:   # 샘플링 속도 확인 및 일치
            raise ValueError("Sample rates of the audio files must be the same.")

        correlation = np.correlate(target, ref, mode='full')   # Cross-Correlation 계산
        lag = np.argmax(correlation) - len(ref)
        return lag, sr_ref
    
    def align_audio(self, reference_file, target_file, output_file):
        lag, sr = self.calculate_time_lag(reference_file, target_file)   # 시간차 계산
        target, _ = librosa.load(target_file, sr=sr)   # 오디오 로드
        aligned_target = np.pad(target, (lag, 0), mode='constant') if lag > 0 else target[-lag:]

        sf.write(output_file, aligned_target, sr)     # 정렬된 오디오 저장
        print(f"Aligned audio saved to {output_file}")

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

    def seperate_speakers(self, audio_file, num_speakers=2):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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") 
        diarization = pipeline(audio_file, num_speakers=num_speakers)
        for result in diarization.itertracks(yield_label=True):
            print(result)