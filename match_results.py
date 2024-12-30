from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import json
import os 

with open(os.path.join('./data', 'diar_chunk_lufs_norm_ibk-poc-meeting_20241220_0-p.json'), mode='r', encoding='utf-8') as f:
    diar_data = json.load(f)

with open(os.path.join('./data/result', 'cstt_lufs_norm_ibk-poc-meeting_20241220_0.json'), mode='r', encoding='utf-8') as f:
    stt_data = json.load(f)

TIME_TOLERANCE = 0.5

def merge_diarization_segments_with_priority(diar_segments):
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

def map_speaker_with_nested_check(diar_segments, stt_segment):
    """STT 구간과 Diarization 구간 비교 후, 중첩된 구간 확인"""
    stt_start, stt_end = stt_segment['start_time'], stt_segment['end_time']
    stt_duration = stt_end - stt_start
    print(f'stt_segment: {stt_segment}')
    candidates = []

    for diar_seg in diar_segments:   # STT 결과값과 겹치는 Diar 구간 탐색 
        diar_start, diar_end = diar_seg['start'], diar_seg['end']
        if stt_start <= diar_end and stt_end >= diar_start:  # 겹침 조건
            candidates.append(diar_seg)
    print(f'candidate: {candidates}')
    if not candidates:
        print("No overlapping segments found.")
        stt_segment['speaker'] = 'Unknown'
        return stt_segment

    for candidate in candidates:
        diar_start, diar_end = candidate['start'], candidate['end']
        nested_segments = [  #  Diar 구간 내에 또 다른 구간 탐색 (0~19 -> 13~14)
            seg for seg in diar_segments
            if seg['start'] >= diar_start and seg['end'] <= diar_end and seg != candidate
        ]
        if nested_segments:   # 또 다른 발화가 있을 때 
            # print(f'nested_seg: {nested_segments}')
            for nested in nested_segments:
                if is_similar(nested, stt_segment):
                    print('true')
                    stt_segment['speaker'] = nested['speaker']
                    return stt_segment
        # 또 다른 발화가 없을 때 (겹치는 후보군들만 탐색)
        max_overlap = 0
        best_speaker = 'Unknown'
        for diar_seg in candidates:
            overlap_start = max(stt_start, diar_seg['start'])
            overlap_end = min(stt_end, diar_seg['end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = diar_seg['speaker']
        stt_segment['speaker'] = best_speaker
        return stt_segment

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

merged_diar_data = merge_diarization_segments_with_priority(diar_data)
with open('./data/updated_diar_with_speakers-merged.json', 'w') as output_file:
    json.dump(merged_diar_data, output_file, ensure_ascii=False, indent=4)

updated_stt_data = []
for stt_segment in stt_data:
    updated_stt_data.append(map_speaker_with_nested_check(merged_diar_data, stt_segment))

# print(updated_stt_data)

# 결과 저장
with open('./data/updated_stt_with_speakers.json', 'w') as output_file:
    json.dump(updated_stt_data, output_file, ensure_ascii=False, indent=4)
print("화자 정보가 추가된 STT 결과가 저장되었습니다.")

def calculate_nonsilent_start(audio_file_path, silence_thresh=-30, min_silence_len=500):
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

print(calculate_nonsilent_start('./data/output/chunk/chunk_lufs_norm_ibk-poc-meeting_20241220_5.wav'))