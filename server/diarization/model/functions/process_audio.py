import os
import json
import logging
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ClusteringDiarizer
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_audio(file_path, num_speakers, config):
    """
    Process the audio files for diarization.

    Args:
        file_path (str): Path to the main audio file.
        reference_audio_paths_with_labels (list): List of (label, path) tuples for reference audios.
        num_speakers (int): Number of speakers to identify.
        config (OmegaConf): Configuration object for the diarizer.
        verify_speaker_model: Speaker verification model.

    Returns:
        str: RTTM content with diarization results.

    Raises:
        Exception: If there are issues during the diarization process.
    """
    # Create a temporary input manifest
    input_manifest = {
        "audio_filepath": file_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None
    }

    # Ensure the data directory exists and save the manifest
    data_dir = os.path.dirname(config.diarizer.manifest_filepath)
    os.makedirs(data_dir, exist_ok=True)
    with open(config.diarizer.manifest_filepath, "w") as f:
        json.dump(input_manifest, f)
        f.write('\n')

    try:
        # Perform diarization
        sd_model = ClusteringDiarizer(cfg=config)
        sd_model.diarize()

        # Get the output RTTM file path
        wav_filename = os.path.basename(file_path)
        rttm_filename = os.path.splitext(wav_filename)[0] + ".rttm"
        rttm_path = os.path.join(
            config.diarizer.out_dir, "pred_rttms", rttm_filename)

        # Read and validate the RTTM file content
        if not os.path.exists(rttm_path):
            raise Exception("RTTM file was not generated.")

        with open(rttm_path, "r") as f:
            rttm_content = f.read()

        if not rttm_content.strip():
            raise Exception(
                "Empty RTTM content, possibly due to silence in audio.")

        return rttm_content
    except Exception as e:
        logger.error(f"Exception in process_audio: {str(e)}")
        raise Exception(f"Error processing audio: {str(e)}")
