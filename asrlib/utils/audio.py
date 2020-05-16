import soundfile as sf
import os
import numpy as np
import wave

from asrlib.utils import base

float32_floor = np.array(np.finfo(float).eps, dtype='float32')
default_audio_sample_rate = 16000
default_audio_bit_num = 16
default_fbank80_config = os.path.join(os.path.dirname(__file__), '../configs/fbank80_config.py')


def frame_num(pcm_lens, sample_rate, frame_length=0.025, frame_step=0.01):
    """
    compute the frame number given the length of signals
    Args:
        pcm_lens: a Tensor, of batch [batch_size]
        sample_rate: python int, sample rate
        frame_length: python float, frame length in second
        frame_step: python float, frame step in second

    Returns:
        a Tensor including the frame number
    """
    frame_length = int(sample_rate * frame_length)
    frame_step = int(sample_rate * frame_step)
    num_frames = (pcm_lens - frame_length + frame_step) // frame_step
    return num_frames


def parse_wav_line(wav_line):
    """
    given a line to describe the wav info and read the wav into stream
    For example:
        "/workspace/data/a.wav" -> read the wav file
        "sox -t wav a.wav -t wav - |" -> run sox command and get the output wav stream
    Args:
        wav_line:
            a wav path, or a sox command or a other command
    Returns:
        a StringIO
    """
    if not base.is_str(wav_line):
        return wav_line

    wav_line = wav_line.strip()
    if wav_line.endswith('|'):
        # this is a command line
        stream = base.run_sox_silence(wav_line.strip('|'),
                                      input_stream=None,
                                      output_stream=base.BytesIO())[0]
        from_type = 'wav'
    else:
        stream = base.BytesIO(open(wav_line, 'rb').read())
        from_type = os.path.splitext(wav_line)[-1]

    # convert speech
    return base.convert_speech(stream, base.BytesIO(),
                               from_type=from_type,
                               to_type='wav',
                               to_bit=default_audio_bit_num,
                               to_sample_rate=default_audio_sample_rate)


def simple_read_wav_to_pcm_float(wav_line):
    pcm_float, _ = sf.read(parse_wav_line(wav_line))
    return pcm_float


def frames_to_seconds(num_frames, frame_length, frame_step):
    """
    compute the time given the frame number
    """
    if num_frames == 0:
        return 0
    return (num_frames - 1) * frame_step + frame_length


def seconds_to_frames(seconds, frame_length, frame_step):
    if seconds == 0:
        return 0
    return int((seconds - frame_length) / frame_step) + 1


def read_wav_to_bits(wav_line):
    f = wave.open(parse_wav_line(wav_line), 'rb')
    bits = f.readframes(f.getnframes())
    return bits


def write_bits_to_wav(bits, output_stream, sample_rate=default_audio_sample_rate, num_channel=1, sample_width=2):
    f = wave.open(output_stream, 'wb')
    f.setnchannels(num_channel)
    f.setsampwidth(sample_width)
    f.setframerate(sample_rate)
    f.writeframes(bits)
    f.close()
