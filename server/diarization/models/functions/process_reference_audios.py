from models.functions.save_temp_audio import save_temp_audio
from models.functions.convert_audio_to_mono_wav import convert_audio_to_mono_wav
import logging
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_reference_audios(reference_audios, labels, temp_files):
    """Process reference audios and return a list of (label, file_path) tuples."""
    reference_audio_paths_with_labels = []
    for ref_bytes, label in zip(reference_audios, labels):
        ref_wav_io = convert_audio_to_mono_wav(ref_bytes)
        temp_ref_file_path = save_temp_audio(
            ref_wav_io, temp_files, "reference")
        reference_audio_paths_with_labels.append(
            (label, temp_ref_file_path))
    logger.info(f"Reference files: {reference_audio_paths_with_labels}")
    return reference_audio_paths_with_labels
