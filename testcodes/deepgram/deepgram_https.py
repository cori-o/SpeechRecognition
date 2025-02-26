from dotenv import load_dotenv
from datetime import datetime
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
import httpx
import json
import os

load_dotenv()
deepgram_api_key = os.getenv('DG_API')
AUDIO_FILE = "./data/전략기획사업단_01.wav"
save_file_name = "deepgram_" + AUDIO_FILE.split('/')[-1].split('.')[0] + '.json'

def main():
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {deepgram_api_key}",
    }
    params = {
        "model": "nova-2",
        "language": "ko",
        "punctuate": False,
        "diarize": False,
        "smart_format": True,
        "no_speech_threshold": 0.1,  # 고급 옵션
        "vad_turnoff": 0.5,         # 고급 옵션
    }

    before = datetime.now()
    with open(AUDIO_FILE, "rb") as audio:
        response = httpx.post(
            url, headers=headers, params=params, content=audio.read(), timeout=300.0
        )

    after = datetime.now()
    parsed_result = response.json()
    print(parsed_result)

    if "utterances" in parsed_result["results"]:
        utterances = parsed_result["results"]["utterances"]
        for idx, utterance in enumerate(utterances, start=1):
            print(f"Utterance {idx}:")
            print(f" - Start Time: {utterance['start']}")
            print(f" - End Time: {utterance['end']}")
            print(f" - Transcript: {utterance['transcript']}")
            print("")
    else:
        print("No utterances found in the response.")

    formatted_segments = []
    for segment in parsed_result["results"]["utterances"]:
        formatted_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "transcript": segment["transcript"],
            "confidence": segment["confidence"]
        })

    with open(save_file_name, "w", encoding="utf-8") as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=4)

    print("")
    difference = after - before
    print(f"time: {difference.seconds}")

if __name__ == "__main__":
    main()