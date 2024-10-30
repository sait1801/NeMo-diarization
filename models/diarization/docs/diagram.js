+-----------------------------------------+
|        TritonPythonModel (model.py)     |
|-----------------------------------------|
| initialize()                            |
|  - Load configurations and models       |
|  - Set diarization settings             |
| execute()                               |
|  - Process requests                     |
|  - Calls diarize_audio()                |
+-----------------------------------------+
                 |
                 |
                 v
+-----------------------------------------+
|        diarize_audio (diarize_audio.py) |
|-----------------------------------------|
| - Calls convert_audio_to_mono_wav()     |
| - Calls process_reference_audios()      |
| - Calls process_audio()                 |
| - Calls cleanup_temp_files()            |
+-----------------------------------------+
                 |
                 |
                 v
+-----------------------------------------+
| process_reference_audios (process_reference_audio.py)|
|-----------------------------------------|
| - Calls save_temp_audio()               |
| - Calls convert_audio_to_mono_wav()     |
+-----------------------------------------+
                 |
                 |
                 v
+-----------------------------------------+
|       process_audio (process_audio.py)  |
|-----------------------------------------|
| - Diarizes using ClusteringDiarizer     |
| - Returns RTTM file content             |
+-----------------------------------------+
                 |
                 |
                 v
+-----------------------------------------+
|        Cleanup (cleanup_temp_files.py)  |
|-----------------------------------------|
| - Deletes temp files                    |
+-----------------------------------------+
