from pydub.effects import high_pass_filter, low_pass_filter
from pyannote.audio import Pipeline
from pydub import AudioSegment
from io import BytesIO
import torch
import noisereduce as nr
import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import subprocess
import librosa
import tempfile
import pickle
import json
import os
import io
import gc
import assemblyai as aai
import requests


class AudioFileProcessor:
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

    def bytesio_to_tempfile(self, audio_bytesio):
        """
        BytesIO 객체를 임시 파일로 저장
        args:
            audio_bytesio (BytesIO): BytesIO 객체
        returns:
            str: 임시 파일 경로.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytesio.getvalue())
            temp_file.flush()
            return temp_file.name

    def file_to_wav(self, file_path, wav_file_path, file_type, sample_rate=None, channels=None, bit_depth=None):
        if file_type == 'pcm':
            with open(pcm_file_path, 'rb') as pcm_file:   # PCM 파일 열기
                pcm_data = pcm_file.read()

            with wave.open(wav_file_path, 'wb') as wav_file:   # WAV 파일 생성
                wav_file.setnchannels(channels)   # 채널 수 (1: 모노, 2: 스테레오)
                wav_file.setsampwidth(bit_depth // 8)   # 샘플 당 바이트 수
                wav_file.setframerate(sample_rate)   # 샘플링 속도
                wav_file.writeframes(pcm_data)   # PCM 데이터 쓰기
        elif file_type == 'm4a':
            audio_file = AudioSegment.from_file(m4a_path, format='m4a')
            wav_path = m4a_path.replace('m4a', 'wav')
            audio_file.export(wav_path, format='wav')


class NoiseHandler: 
    '''
    음성 파일에서 노이즈를 제거한다.
    '''
    def remove_background_noise(self, input_file, output_file=None, prop_decrease=None):
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
        # noise_profile = y[:sr]   # 배경 잡음 프로파일 추출 (초기 1초)
        # y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile, prop_decreaese=prop_decrease)   # 잡음 제거
        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
        del y
        gc.collect()

        if output_file: 
            sf.write(output_file, y_denoised, sr)
            print(f"Saved denoised audio to {output_file}")
        wav_buffer = io.BytesIO()   # 메모리 내 WAV 파일 생성
        sf.write(wav_buffer, y_denoised, sr, format='WAV')
        wav_buffer.seek(0)   # 파일 포인터를 처음으로 이동
        del y_denoised
        gc.collect()
        return wav_buffer

    def remove_noise_with_ffmpeg(self, input_file, output_file=None, model_file="rnnoise_model.rnnn"):
        """
        FFmpeg을 사용한 배경 잡음 제거.
        Args:
            input_file: 입력 오디오 파일 (str, os.PathLike, io.BytesIO).
            output_file: 출력 파일 경로 (str). 지정하지 않으면 BytesIO로 반환.
            model_file: RNNoise 모델 파일 경로 (str).
        Returns:
            io.BytesIO: 잡음 제거된 오디오 데이터를 담은 BytesIO 객체 (output_file이 None일 때).
        """
        # 입력 파일 처리
        if isinstance(input_file, io.BytesIO):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                temp_input.write(input_file.getvalue())
                temp_input.flush()
                input_source = temp_input.name
        elif isinstance(input_file, (str, os.PathLike)):
            input_source = input_file
        else:
            raise ValueError("input_file must be a file path or a BytesIO object")

        try:
            # FFmpeg 명령어 구성
            command = [
                "ffmpeg",
                "-y",
                "-i", input_source,
                "-af", f"arnndn=m={model_file}",
                "-f", "wav",
            ]

            if output_file:
                command.append(output_file)
                subprocess.run(command, check=True)
                print(f"Noise removed audio saved to {output_file}")
                return output_file
            else:
                command.append("pipe:1")  # 표준 출력으로 결과 전달
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
                return io.BytesIO(stdout)  # BytesIO로 반환
        finally:
            # 임시 파일 정리
            if isinstance(input_file, io.BytesIO) and 'temp_input' in locals():
                os.unlink(temp_input.name)

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

    def filter_audio_with_ffmpeg(self, input_file, high_cutoff=50, low_cutoff=4000, output_file=None):
        """
        FFmpeg을 사용한 오디오 필터링 (고역대, 저역대).
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
                "-y",
                "-i", input_source,
                "-af", f"highpass=f={high_cutoff},lowpass=f={low_cutoff}",
                "-f", "wav",
                "pipe:1" if not output_file else output_file
            ]
            if output_file:
                subprocess.run(command, check=True)
                print(f"Filtered audio saved to {output_file}")
                
            # stdout으로 데이터를 읽고 BytesIO로 반환
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
            return io.BytesIO(stdout)  # BytesIO로 반환
        finally:
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

    def normalize_audio_pydub(self, audio_file_path, audio_file_name, target_dbfs=-14):
        audio = AudioSegment.from_file(os.path.join(audio_file_path, audio_file_name))
        
        # RMS(root mean square) 기반 볼륨 정규화
        normalized_audio = audio.apply_gain(-audio.dBFS)

        # 정규화된 오디오 저장
        output_file = os.path.join(audio_file_path, "pydub_nr_" + audio_file_name)
        normalized_audio.export(output_file, format="wav")
        return output_file

        # RMS 기반 볼륨 정규화 + 클리핑 방지
        change_in_dBFS = target_dbfs - audio.dBFS
        if change_in_dBFS < 0:  # 음량 감소
            normalized_audio = audio.apply_gain(change_in_dBFS)
        else:
            # 음량 증가 시 클리핑 방지
            normalized_audio = audio.apply_gain(change_in_dBFS).clip(min=-1.0, max=1.0)


    def normalize_audio_lufs(self, audio_input, target_lufs=-14.0, output_file=None):
        """
        LUFS 기반 오디오 정규화
        """
        #print(audio_input)
        if isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            data, rate = sf.read(audio_input)
        else:
            data, rate = sf.read(audio_input)

        # 현재 LUFS 계산 및 정규화
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)

        if output_file:
            sf.write(output_file, loudness_normalized_audio, rate)
            print(f"Saved normalized audio to {output_file}")
            return output_file
        else:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, loudness_normalized_audio, rate, format='WAV')
            wav_buffer.seek(0)
            return wav_buffer
            

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
    def set_pyannotate(self, hf_api):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_api = hf_api 
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=self.hf_api)
        self.pipeline.to(self.device)

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

    def _process_local_diarization(self, audio_file, max_speakers):
        """
        perform local speaker diarization.
        Parameters:
            audio_file: str
                Path to the audio file.
            num_speakers: int or None
                Number of speakers to use for diarization.
        returns:
            list
                A list of processed diarization results.
        """
        try:
            diarization = self.pipeline(audio_file, num_speakers=None)
        except Exception as e:
            print(f"[ERROR] Diarization failed: {e}")
            return []

        speaker_durations = {}     # Calculate total durations for each speaker
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + (turn.end - turn.start)

        top_speakers = sorted(speaker_durations, key=speaker_durations.get, reverse=True)[:max_speakers]    # Select top N speakers based on duration
        filtered_diarization = []    # Filter diarization results to include only top speakers
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker in top_speakers:
                filtered_diarization.append((turn.start, turn.end, speaker))

        results = []   # Convert results
        for start, end, speaker in filtered_diarization:
            results.append(self.convert_segments((start, end, speaker)))
        return results

    def seperate_speakers(self, data_p, audio_file, local=True, num_speakers=None, save_path=None, file_name=None):
        """
        화자 분리 실행 및 결과 저장.
        args:
            audio_file (str or BytesIO): 입력 오디오 파일 경로 또는 BytesIO 객체.
            save_path (str): 결과 저장 경로.
            file_name (str): 저장할 파일 이름.
            num_speakers (int, optional): 예상 화자 수. 기본값은 Pyannote에서 자동 감지.
        """
        if isinstance(audio_file, io.BytesIO):   # 입력 데이터 형식 확인 및 변환
            audio_file = data_p.bytesio_to_tempfile(audio_file)
        
        results = []
        if local:
            try:   # Pyannote Pipeline 초기화
                diarization = self.pipeline(audio_file)
            except Exception as e:
                print(f"[ERROR] Diarization failed: {e}")
                return
            for result in diarization.itertracks(yield_label=True):  # result: (<Segment>, _, speaker)
                if int(result[-1].split('_')[-1]) > num_speakers - 1:
                    continue 
                converted_info = self.convert_segments(result)
                results.append(converted_info)
        else:
            pass 
        if save_path != None:    # 저장 경로 확인 및 결과 저장
            os.makedirs(save_path, exist_ok=True)
            save_file_path = os.path.join(save_path, file_name)
            with open(save_file_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {save_file_path}")
        return results