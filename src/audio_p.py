from pydub.effects import high_pass_filter, low_pass_filter
from datetime import datetime, timedelta
from datasets import Dataset
from pyannote.audio import Pipeline
from collections import Counter
from pydub import AudioSegment
from io import BytesIO
import pyloudnorm as pyln
import noisereduce as nr
import soundfile as sf
import pandas as pd
import numpy as np 
import subprocess
import tempfile
import librosa
import torch
import pickle
import wave
import json
import re
import io
import os


class DataProcessor:
    def data_to_df(self, dataset, columns):
        if isinstance(dataset, list):
            return pd.DataFrame(dataset, columns=columns)
       
    def df_to_hfdata(self, df):
        return Dataset.from_pandas(df)

    def merge_data(self, df1, df2, how='inner', on=None):
        return pd.merge(df1, df2, how='inner')

    def filter_data(self, df, col, val):
        return df[df[col]==val].reset_index(drop=True)
    
    def remove_keywords(self, df, col, keyword=None, exceptions=None):
        if exceptions != None: 
            if keyword != None:
                # pattern = r'(?<![\w가-힣])(?:' + '|'.join(map(re.escape, keyword)) + r')(?![\w가-힣])'
                pattern = re.compile(r'(?<![\w가-힣])(' + '|'.join(map(re.escape, val)) + r')(?=[^가-힣]|$)')
            else:
                pattern = r'(?<![\w가-힣])(\S*주)(?![\w가-힣])'    # 테마주 같은 함정 증권 종목 제거 
            mask = df[col].str.contains(pattern, na=False) & ~df[col].str.contains('|'.join(map(re.escape, exceptions)), na=False)
            df = df[~mask]
            return df.reset_index(drop=True)
        else: 
            keyword_idx = df[df[col].str.contains(keyword, na=False)].index 
            df.drop(keyword_idx, inplace=True)
            return df.reset_index(drop=True)
        
    def save_results_to_pickle(self, result, output_file):
        with open(output_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Results saved to {output_file}")

    def load_results_from_pickle(self, input_file):
        with open(input_file, "rb") as f:
            result = pickle.load(f)
        print(f"Results loaded from {input_file}")
        return result


class TextProcessor:
    def count_pattern(self, text, patterns):
        cnt = 0 
        for pattern in sorted(patterns, reverse=True):
            if pattern in text: 
                cnt += 1 
                text = text.replace(pattern, '', 1)
        return cnt 
                   
    def remove_duplications(self, text):
        '''
        줄바꿈 문자를 비롯한 특수 문자 중복을 제거합니다.
        '''
        text = re.sub(r'(\n\s*)+\n+', '\n\n', text)    # 다중 줄바꿈 문자 제거
        text = re.sub(r"\·{1,}", " ", text)    
        return re.sub(r"\.{1,}", ".", text)
        
    def remove_patterns(self, text, pattern):
        '''
        입력된 패턴을 제거합니다. 
        pattern:
        r'^\d+\.\s*': "숫자 + ." 
        r"(뉴스|주식|정보|분석)$": 삼성전자뉴스 -> 삼성전자
        '''
        return re.sub(pattern, '', text)
    
    def check_expr(self, expr, text):
        '''
        expr 값이 text에 있는지 검사합니다. 있다면 True를, 없다면 False를 반환합니다. 
        '''
        return bool(re.search(expr, text))
    
    def get_val(self, val, text):
        '''
        expr 값이 text에 있으면 반환합니다.  
        '''
        if isinstance(val, str):
            return re.search(rf'\b{re.escape(val)}\b')
        elif isinstance(val, list):
            return [re.search(rf'\b{re.escape(v)}\b') for v in val]
    
    def get_val_with_indices(self, val, text):
        '''
        val 값이 text에 있으면 시작과 끝 위치 정보와 함께 값을 반환합니다.
        '''
        found_stocks = []
        if isinstance(val, str):
            pattern = rf'(^|[^a-zA-Z0-9가-힣]){re.escape(val)}($|[^a-zA-Z0-9가-힣])'
            matches = list(re.finditer(pattern, text))
            for match in matches:
                # 매칭된 텍스트에서 실제 단어의 시작과 끝 위치 조정
                start = match.start() + (1 if match.group(1) else 0)
                end = match.end() - (1 if match.group(2) else 0)
                found_stocks.append((val, start, end))
            return found_stocks
        elif isinstance(val, list):
            pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, val)) + r')\b')
            matches = [(match.group(), match.start(), match.end()) for match in pattern.finditer(text)]
            return matches 
            
    def check_l2_threshold(self, txt, threshold, value):
        '''   
        threshold 보다 값이 높은 경우, 모르는 정보로 간주합니다. 
        '''
        print(f'Euclidean Distance: {value}, Threshold: {threshold}')
        return "모르는 정보입니다." if value > threshold else txt
    

class TimeProcessor:
    def get_previous_day_date(self):
        '''
        전일 연도, 월, 일을 반환합니다.   
        returns: 
        20240103
        '''
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        return str(yesterday.year), str(yesterday.month).zfill(2), str(yesterday.day).zfill(2)

    def get_current_date(self):
        '''
        현재 연도, 월, 일을 반환합니다.
        '''
        now = datetime.now()
        return str(now.year), str(now.month).zfill(2), str(now.day).zfill(2)

    def get_current_time(self):
        return datetime.now()

    def is_similar(diar_seg, stt_seg):
        '''
        두 세그먼트 간 겹치는 길이와 발화 시간이 유사한지 검사
        '''
        diar_start, diar_end = diar_seg['start'], diar_seg['end']
        stt_start, stt_end = stt_seg['start_time'], stt_seg['end_time']

        diar_duration = diar_end - diar_start
        stt_duration = stt_end - stt_start

        TIME_TOLERANCE = 1.5   # 허용 오차(초)

        # 겹치는 구간 계산
        overlap_start = max(diar_start, stt_start)
        overlap_end = min(diar_end, stt_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        # print(diar_seg, stt_seg)
        print(f'overlap duration: {overlap_duration}, abs: {abs(diar_duration - stt_duration)}')

        # 조건: 겹침이 충분히 길고, 발화 시간도 비슷해야 함
        if overlap_duration > 0.5 and abs(diar_duration - stt_duration) < TIME_TOLERANCE:
            return True
        else:
            return False

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

    def m4a_to_wav(self, m4a_path):
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
        if output_file: 
            sf.write(output_file, y_denoised, sr)
            print(f"Saved denoised audio to {output_file}")
        
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
        input_source = None   # 변수 초기화
        temp_files = []   # 임시 파일을 저장할 리스트
        try:
            if isinstance(input_file, AudioSegment):   # AudioSegment 객체 처리
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    input_file.export(temp_input, format="wav")   # AudioSegment -> WAV 변환
                    temp_input.flush()
                    input_source = temp_input.name
                    temp_files.append(temp_input.name)   # 임시 파일 관리
            elif isinstance(input_file, io.BytesIO):   # BytesIO 객체 처리
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    temp_input.write(input_file.getvalue())
                    temp_input.flush()
                    input_source = temp_input.name
                    temp_files.append(temp_input.name)   # 임시 파일 관리
            elif isinstance(input_file, (str, os.PathLike)):   # 파일 경로 처리
                input_source = input_file
            else:
                raise ValueError("Invalid input_file type. Must be AudioSegment, file path, or BytesIO object.")

            if input_source is None:
                raise RuntimeError("Failed to determine input source.")

            command = [   # FFmpeg 명령 실행
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
            for temp_file in temp_files:   # 임시 파일 삭제
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


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

    def rename_speaker(self, result):
        '''
        화자 분리 결과에서 발화량이 많은 순으로 화자 번호 재부여
        초과하는 화자의 발화를 제거
        '''
        speaker_counts = Counter(entry['speaker'] for entry in result)
        print(f'speaker_counts: {speaker_counts}')
        sorted_speakers = [speaker for speaker, _ in speaker_counts.most_common()]
        speaker_mapping = {old_speaker: f"SPEAKER_{i:02d}" for i, old_speaker in enumerate(sorted_speakers)}

        filtered_result = []
        for entry in result:
            new_speaker = speaker_mapping[entry['speaker']]
            entry['speaker'] = new_speaker
            filtered_result.append(entry)
        return filtered_result
    
    def filter_speaker_segments(self, segments):
        '''
        겹치는 발화 제거 (0~10, 3~5 -> 3~5 제거)
        '''
        filtered_segments = []
        for i, seg in enumerate(segments):
            if i > 0 and seg["speaker"] != segments[i - 1]["speaker"]:   # 이전 발화와 확인
                prev_seg = segments[i - 1]
                if prev_seg["start"] <= seg["start"] and seg["end"] <= prev_seg["end"]:
                    continue
            filtered_segments.append(seg)
        return filtered_segments
    
    def calc_speak_duration(self, segments, speaker):
        speak_time = 0; cnt = 0
        for seg in segments:
            if seg['speaker'] == speaker:
                speak_duration = seg['end'] - seg['start']
                speak_time += speak_duration 
                cnt += 1
        print(f'{speaker}: {round((speak_time / cnt), 2)}초')

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

    def merge_diarization_segments_with_priority(self, diar_segments):
        """Diarization 구간 병합 시 추임새와 긴 발화 분리"""
        merged_segments = []
        FILLER_THRESHOLD = 2.0   # 추임새로 간주할 최대 길이 (초)
        for seg in diar_segments:
            if not merged_segments:
                merged_segments.append(seg)
                continue
            last_seg = merged_segments[-1]

            # 화자가 다르고, 추임새로 간주되는 경우 (추임새는 긴 발화 구간에 병합하지 않음)
            if last_seg['speaker'] != seg['speaker'] and (seg['end'] - seg['start']) <= FILLER_THRESHOLD:
                merged_segments.append(seg)
            elif last_seg['speaker'] == seg['speaker'] and last_seg['end'] >= seg['start']:
                last_seg['end'] = max(last_seg['end'], seg['end'])   # 같은 화자의 연속된 발화 병합
            else:
                merged_segments.append(seg)   # 새로운 화자 구간 추가
        return merged_segments
            
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
            print(f'start diarization')
            diarization = self.pipeline(audio_file, num_speakers=None)
            for result in diarization.itertracks(yield_label=True):   # result: (<Segment>, _, speaker)
                converted_info = self.convert_segments(result)
                results.append(converted_info)
        else:
            pass
        filtered_result = self.filter_speaker_segments(results)
        merged_result = self.merge_diarization_segments_with_priority(filtered_result)
        diar_result = self.rename_speaker(merged_result)
        
        if save_path != None:    # 저장 경로 확인 및 결과 저장
            os.makedirs(save_path, exist_ok=True)
            save_file_path = os.path.join(save_path, file_name)
            with open(save_file_path, "w") as f:
                json.dump(diar_result, f, indent=4)
            print(f"Results saved to {save_file_path}")
        return results


class ETC:
    '''
    당장 쓰이지 않는 메서드 정의
    '''
    def get_model_response(self, df, user_id, query):
        qa_pairs = []
        current_question = None
        question_time = None

        user_df = df[df['user_id'] == user_id].reset_index(drop=True)
        user_df = user_df.sort_values('date')
        # user_df.to_csv('./debug_user.csv', index=False)
        
        for i, row in user_df.iterrows():   # 질문-응답 매칭 루프
            if row['q/a'] == 'Q' and row['content'] == query:
                current_question = row['content']
                question_time = row['date']
            elif row['q/a'] == 'A' and current_question is not None:
                response_time = row['date']   # 질문에 대한 응답을 기록
                time_diff = response_time - question_time   # 시간 차이가 많이 나지 않는 경우에만 질문과 응답을 매칭
                if time_diff.seconds < 300:  # 5분 이내
                    qa_pairs.append({
                        'question': current_question,
                        'answer': row['content'],
                        'question_time': question_time,
                        'answer_time': response_time
                    })
                current_question = None   # 응답 처리 후 질문 초기화
                question_time = None
        return qa_pairs