from pydub import AudioSegment
import io


def convert_audio_to_mono_wav(audio_bytes):
    """
    Convert audio bytes to mono WAV format.

    Args:
        audio_bytes (bytes): Input audio file content.

    Returns:
        io.BytesIO: BytesIO object containing the converted WAV audio.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    # Normalize audio to -20 dBFS
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)
    # Export the audio to a BytesIO object in WAV format
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io
