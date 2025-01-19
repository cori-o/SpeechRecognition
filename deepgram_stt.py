# Copyright 2023-2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT
from deepgram.utils import verboselogs
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
    config: DeepgramClientOptions = DeepgramClientOptions(
        verbose=verboselogs.SPAM,
    )
    deepgram: DeepgramClient = DeepgramClient(deepgram_api_key, config)

    # STEP 2 Call the transcribe_file method on the rest class
    with open(AUDIO_FILE, "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    options: PrerecordedOptions = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        language='ko',
        punctuate=False,
        diarize=False,
    )
    before = datetime.now()
    response = deepgram.listen.rest.v("1").transcribe_file(
        payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
    )
    after = datetime.now()
#    json_result = response.to_json(indent=4, ensure_ascii=False)
    parsed_result = response.to_dict()

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