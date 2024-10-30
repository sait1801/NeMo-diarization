import tempfile
from pydub import AudioSegment
from models.functions.convert_audio_to_mono_wav import convert_audio_to_mono_wav
from models.functions.process_audio import process_audio
from models.functions.cleanup_temp_files import cleanup_temp_files
from models.functions.verify_speakers import verify_speakers
from models.functions.save_temp_audio import save_temp_audio
from models.functions.process_reference_audios import process_reference_audios


def diarize_audio(file_bytes, reference_audios, labels, num_speakers, verify_speaker_model, config):

    if not file_bytes:
        raise Exception("Main audio file is missing.")

    if len(reference_audios) != len(labels):
        raise Exception(
            "The number of reference audios and labels must match.")

    temp_files = []

    try:
        # Convert and save main audio
        main_audio_wav_io = convert_audio_to_mono_wav(file_bytes)
        main_audio = AudioSegment.from_file(main_audio_wav_io)

        temp_main_file_path = save_temp_audio(
            main_audio_wav_io, temp_files, "main")

        # Process reference audios
        reference_audio_paths_with_labels = process_reference_audios(
            reference_audios, labels, temp_files)

        # Perform diarization
        rttm_content = process_audio(
            file_path=temp_main_file_path,
            reference_audio_paths_with_labels=reference_audio_paths_with_labels,
            num_speakers=num_speakers,
            config=config,
            verify_speaker_model=verify_speaker_model
        )

        # Call verify_speakers to handle chunk verification
        verification_results = verify_speakers(
            main_audio,
            rttm_content,
            reference_audio_paths_with_labels,
            verify_speaker_model,
            temp_files
        )

        return verification_results

    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)
