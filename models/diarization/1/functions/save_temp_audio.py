import tempfile
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_temp_audio(audio_wav_io, temp_files, prefix):
    """Save audio to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_wav_io.read())
        temp_file_path = temp_file.name
        temp_files.append(temp_file_path)
        logger.info(
            f"{prefix.capitalize()} audio saved to {temp_file_path}")
        return temp_file_path
