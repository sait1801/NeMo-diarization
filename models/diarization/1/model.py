import os
import json
import wget
import numpy as np
import logging
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from typing import List
import triton_python_backend_utils as pb_utils
from models.functions.diarize_audio import diarize_audio

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


################################################################
################### Main Triton Backend Code ###################
################################################################
class TritonPythonModel:
    def initialize(self, args):
        # Model configuration
        self.model_config = json.loads(args['model_config'])

        # Load models
        logger.info("Loading models...")

        # Load the speaker verification model
        self.verify_speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "ecapa_tdnn"
        )

        # Determine model names for VAD and speaker embedding
        self.pretrained_vad = 'vad_marble'
        self.pretrained_speaker_model = 'titanet_large'

        # Define output directories
        self.ROOT = os.getcwd()
        self.data_dir = '/data'  # Absolute path outside the /models directory
        os.makedirs(self.data_dir, exist_ok=True)

        # Download and load the diarization config file
        self.MODEL_CONFIG = os.path.join(
            self.data_dir, 'diar_infer_general.yaml')
        if not os.path.exists(self.MODEL_CONFIG):
            config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml"
            logger.info("Downloading model config...")
            self.MODEL_CONFIG = wget.download(config_url, self.data_dir)

        # Load and adjust the config file
        self.config = OmegaConf.load(self.MODEL_CONFIG)

        # Set clustering parameters
        self.config.diarizer.clustering.parameters.max_rp_threshold = 0.05
        self.config.diarizer.clustering.parameters.max_num_speakers = 5
        self.config.num_workers = 0

        # Set output directory
        self.output_dir = '/outputs'  # Absolute path outside the /models directory

        # Configure paths and Diarization settings
        self.config.diarizer.manifest_filepath = os.path.join(
            self.data_dir, 'input_manifest.json')
        self.config.diarizer.out_dir = self.output_dir
        self.config.diarizer.speaker_embeddings.model_path = self.pretrained_speaker_model
        self.config.diarizer.oracle_vad = False
        self.config.diarizer.clustering.parameters.oracle_num_speakers = True

        # Configure VAD settings
        self.config.diarizer.vad.model_path = self.pretrained_vad
        self.config.diarizer.vad.parameters.onset = 0.5
        self.config.diarizer.vad.parameters.offset = 0.3

        logger.info("Model initialization complete.")

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extract inputs from the request
            file_input = pb_utils.get_input_tensor_by_name(
                request, "file").as_numpy()[0]
            reference_audios_input = pb_utils.get_input_tensor_by_name(
                request, "reference_audios").as_numpy()
            labels_input = pb_utils.get_input_tensor_by_name(
                request, "labels").as_numpy()
            num_speakers_input = pb_utils.get_input_tensor_by_name(
                request, "num_speakers").as_numpy()[0]

            # Convert inputs to appropriate formats
            file_bytes = file_input
            reference_audios = [ref for ref in reference_audios_input]
            labels = [label.decode('utf-8') for label in labels_input]
            num_speakers = int(num_speakers_input)

            # Perform diarization
            try:
                rttm_content = diarize_audio(
                    file_bytes=file_bytes,
                    reference_audios=reference_audios,
                    labels=labels,
                    config=self.config,
                    num_speakers=num_speakers,
                    verify_speaker_model=self.verify_speaker_model
                )

                # Prepare output tensor
                rttm_tensor = pb_utils.Tensor(
                    "rttm_content",
                    np.array([rttm_content.encode('utf-8')], dtype=np.bytes_)
                )

                # Create response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[rttm_tensor]
                )
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                )

            responses.append(inference_response)

        return responses
