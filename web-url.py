from flask import Flask, send_file, request, jsonify, Response
from src import NoiseHandler, VoiceEnhancer, VoiceSeperator, SpeakerDiarizer
from src import DBConnection, PostgresDB, TableEditor
from src import DataProcessor
from src import WhisperSTT
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
import argparse
import requests
import time
import json
import os
import gc

app = Flask(__name__)

@app.route('/stt/', methods=['GET'])
def get_audio_file():
    # 오디오 파일의 전체 경로 확인
    # print(f'{os.path.join(file_path, file_name}')
    file_name = request.args.get('audio_file_name')
    try:
        return send_file(os.path.join(file_path, file_name), mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/stt/result', methods=['POST'])
def webhook():
    '''
    data: {
        "jobId": "bd7e97c9-0742-4a19-bd5a-9df519ce8c74",
        "status": "succeeded",
        "output": {
            "diarization": [
                { "start": 1.2,
                  "end": 3.4,
                  "speaker": "SPEAKER_01" },
                ...
            ]
        }
    }
    '''
    data = request.json
    diar_result = data['output']['diarization']
    print(f'diar result: {diar_result}')
    filtered_result = []
    for result in diar_result:
        if int(result['speaker'].split('_')[-1]) > participant_cnt - 1:
            continue 
        filtered_result.append(result)
    with open(os.path.join(file_path, audio_file_name.split('.')[0] + '.json'), 'w', encoding='utf-8') as f:
        json.dump(filtered_result, f, ensure_ascii=False, indent=4)

    stt_result = stt_module.process_segments_with_whisper(data_p, new_file_name, filtered_result)
    meeting_id = int(audio_file_name.split('_')[1])
    print(f'STT result: {stt_result}, meeting id: {meeting_id}')
    with open(stt_result_path, 'w', encoding='utf-8') as f:
        json.dump(stt_result, f, ensure_ascii=False, indent=4)

    table_editor.edit_poc_conf_tb(task='update', table_name='ibk_poc_conf', data=meeting_id, val=db_stt_result_path)
    return jsonify({"status": "received"}), 200


@app.post('/run')
def run_python_code():
    global participant_cnt 
    global file_path   # /ibk/meeting_records
    global data_p   
    global stt_module
    global new_file_name   # /ibk/meeting_records/ibk-poc-meeting1.wav
    global audio_file_name   # ibk-poc-meeting1.wav
    global stt_result_path   # /ibk/meeting_records/stt_result/ibk-poc-meeting1.json
    global db_stt_result_path   
    global table_editor    
    global stt_module

    load_dotenv()
    data = request.get_json()
    openai_api_key = os.getenv('OPENAI_API')
    pyannot_api_key = os.getenv('PA_API')
    
    file_name = data['file_name']    # /ibk/meeting_records/ibk-poc-meeting1.wav
    file_path = '/ibk/meeting_records/'
    new_file_name = file_path + file_name.split('/')[-1]
    audio_file_name = new_file_name.split('/')[-1]
    participant_cnt = data['participant'] 

    pyannotate_url = "https://api.pyannote.ai/v1/diarize"
    file_url = f"https://ibkpoc.fingerservice.co.kr/stt/?audio_file_name={audio_file_name}"
    webhook_url = "https://ibkpoc.fingerservice.co.kr/stt/result"
    stt_result_path = '/ibk/meeting_records/stt_result/' + audio_file_name.replace('.wav', '.json')
    db_stt_result_path = '/home/jsh0630/meeting_records/stt_result/' + audio_file_name.replace('.wav', '.json')
    
    data_p = DataProcessor()
    stt_module = WhisperSTT(openai_api_key)
    stt_module.set_client()

    with open(os.path.join('./config', 'db_config.json')) as f:
        db_config = json.load(f)
    db_conn = DBConnection(db_config)
    db_conn.connect()
    postgres = PostgresDB(db_conn)
    table_editor = TableEditor(db_conn)

    headers = {
        "Authorization": f"Bearer {pyannot_api_key}"
    }
    data = {
        "webhook": webhook_url,
        "url": file_url
    }
    external_response = requests.post(pyannotate_url, headers=headers, json=data)
    flask_response = Response(
        response=external_response.content,   # 응답 본문
        status=external_response.status_code,    # 상태 코드
        content_type=external_response.headers.get('Content-Type')  # Content-Type 헤더
    )
    return flask_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
