from flask import Flask, send_file, request, jsonify, Response
from src import NoiseHandler, SpeakerDiarizer, AudioFileProcessor, TimeProcessor, WhisperSTT, ResultMapper
from src import DBConnection, TableEditor, PostgresDB
from src import LLMOpenAI
from dotenv import load_dotenv
import requests
import time
import logging
import tempfile
import json
import os

app = Flask(__name__)    

logger = logging.getLogger('stt_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('stt-result.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

load_dotenv()
openai_api_key = os.getenv('OPENAI_API')
with open(os.path.join('./config', 'db_config.json')) as f:
    db_config = json.load(f)

with open(os.path.join('./config', 'llm_config.json')) as f:
    llm_config = json.load(f)
    
db_conn = DBConnection(db_config)
db_conn.connect()
postgres = PostgresDB(db_conn)
table_editor = TableEditor(db_conn)

diar_module = SpeakerDiarizer()
audio_p = AudioFileProcessor()
noise_handler = NoiseHandler()
time_p = TimeProcessor()

stt_module = WhisperSTT(openai_api_key)
stt_module.set_client()
stt_module.load_word_dictionary('/ibk/config/word_dict.json')
'''
openai_client = LLMOpenAI(llm_config, openai_api_key)
openai_client.set_generation_config()
openai_client.set_grammer_guideline()'''
resultMapper = ResultMapper()

@app.route('/stt/', methods=['GET'])
def get_audio_file():
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
    api_result = data['output']['diarization']
    diar_result = diar_module.rename_speaker(api_result, participant_cnt)
    with open(os.path.join(diar_result_path, 'diar_' + audio_file_name.split('.')[0] + '.json'), 'w', encoding='utf-8') as f:
        json.dump(diar_result, f, ensure_ascii=False, indent=4)
    return jsonify({"status": "received"}), 200

@app.post('/run')
def run_python_code():
    global participant_cnt 
    global file_path    # /ibk/meeting_records
    global audio_p
    global new_file_name     # /ibk/meeting_records/meeting_302_2025-01-09-16-25-30.wav
    global audio_file_name     # meeting_302_2025-01-09-16-25-30.wav
    global diar_result_path      # /ibk/meeting_records/stt_result/meeting_302_2025-01-09-16-25-30.json
    
    load_dotenv()
    data = request.get_json()
    pyannot_api_key = os.getenv('PA_API')
    
    file_name = data['file_name']     # /jsh0630/meeting_records/meeting_302_2025-01-09-16-25-30.wav
    participant_cnt = data['participant']

    file_path = '/ibk/meeting_records/'
    diar_result_path = '/ibk/meeting_records/diar_result/'
    new_file_name = file_path + file_name.split('/')[-1]    # meeting_302_2025-01-09-16-25-30.wav
    audio_file_name = new_file_name.split('/')[-1]   # /ibk/meeting_records/meeting_302_2025-01-09-16-25-30.wav

    pyannotate_url = "https://api.pyannote.ai/v1/diarize"
    file_url = f"https://ibkpoc.fingerservice.co.kr/stt/?audio_file_name={audio_file_name}"
    webhook_url = "https://ibkpoc.fingerservice.co.kr/stt/result"
    headers = { "Authorization": f"Bearer {pyannot_api_key}" }
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

@app.post('/run-stt')
def run_stt_code():
    data = request.get_json()
    file_name = data['file_name']     # /jsh0630/meeting_records/meeting_302_2025-01-09-16-25-30.wav

    file_path = '/ibk/meeting_records/'
    new_file_name = file_path + file_name.split('/')[-1]    # /ibk/meeting_records/meeting_302_2025-01-09-16-25-30.wav
    audio_file_name = new_file_name.split('/')[-1]   # meeting_302_2025-01-09-16-25-30.wav
    meeting_id = int(audio_file_name.split('_')[1])
    print(f'meeting ID: {meeting_id}')
    chunk_length = 270
    audio_chunk = audio_p.audio_chunk(new_file_name, chunk_length=chunk_length)   # 270 sec 
    for idx, chunk in enumerate(audio_chunk):
        chunk_offset = idx * chunk_length   # 청크의 시작 시간 오프셋 계산
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            chunk.export(temp_audio_file.name, format="wav")
            temp_audio_path = temp_audio_file.name
        stt_module.transcribe_text(audio_p, temp_audio_path, meeting_id, table_editor, chunk_offset)
        if idx == 0:
            table_editor.edit_poc_conf_tb(task='update', table_name='ibk_poc_conf', data=meeting_id)
    return jsonify({"status": "received"}), 200

@app.post('/map-result')
def merge_result_code():
    data = request.get_json()
    meeting_id = data['meetingId']
    time.sleep(30)
    with open(os.path.join(diar_result_path, 'diar_' + audio_file_name.split('.')[0] + '.json'), 'r', encoding='utf-8') as f:
        diar_results = json.load(f)
    stt_results = postgres.get_conf_log_data(meeting_id)
    for stt_result in stt_results: 
        resultMapper.map_speaker_with_nested_check(time_p, stt_result, diar_results, meeting_id=meeting_id, table_editor=table_editor)
    return jsonify({"status": "received"}), 200

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)