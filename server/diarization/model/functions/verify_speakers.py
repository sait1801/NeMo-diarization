import tempfile
from pydub import AudioSegment


def verify_speakers(main_audio, rttm_content, reference_audio_paths_with_labels, verify_speaker_model, temp_files):
    """
    Verify the speaker in each chunk of the main audio file against reference audios.

    Args:
        main_audio (AudioSegment): Main audio loaded as a pydub AudioSegment.
        rttm_content (str): RTTM content with diarization results.
        reference_audio_paths_with_labels (list): List of (label, path) tuples for reference audios.
        verify_speaker_model: Preloaded speaker verification model.
        temp_files (list): List to store paths of temporary files for cleanup.

    Returns:
        dict: Dictionary where each chunk path has verification results.
    """
    verification_results = {}

    for line in rttm_content.splitlines():
        tokens = line.split()
        if tokens[0] != "SPEAKER":
            continue

        start_time = float(tokens[3])
        duration = float(tokens[4])
        speaker_label = tokens[7]

        # Create the audio chunk based on the start time and duration
        chunk_audio = main_audio[start_time *
                                 1000:(start_time + duration) * 1000]

        # Save chunk to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
            temp_chunk_file_path = temp_chunk_file.name
            chunk_audio.export(temp_chunk_file, format="wav")
            temp_files.append(temp_chunk_file_path)

        # Verify speakers for each chunk
        verification_results[temp_chunk_file_path] = {}
        for label, ref_audio_path in reference_audio_paths_with_labels:
            result = verify_speaker_model.verify_speakers(
                temp_chunk_file_path, ref_audio_path)
            verification_results[temp_chunk_file_path][label] = result

        # Output verification results for this chunk
        print(
            f"Verification results for chunk '{temp_chunk_file_path}': {verification_results[temp_chunk_file_path]}")

    return verification_results
