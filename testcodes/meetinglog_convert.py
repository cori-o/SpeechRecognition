import json 
import os 

file_name = '20250225.json'
output_file_name = "simple_" + file_name
with open(os.path.join('./meeting_records', 'output_20250225.json'), "r", encoding="utf-8") as f:
    meeting_log = json.load(f)

simple_log = [{"text": entry["text"], "speaker_info": entry["speaker_info"]} for entry in meeting_log]
with open(os.path.join('./meeting_records', output_file_name), "w", encoding="utf-8") as output_file:
    json.dump(simple_log, output_file, ensure_ascii=False, indent=4)