import os
import sys
import numpy as np
import subprocess
from absl import logging
from collections import OrderedDict

try:
    from Queue import Queue
except ImportError:
    from queue import Queue


from asrlib.utils.reader import SaveOpen, stream2str, load_vocab, read_txt_to_dict
from asrlib.utils.reader import StringIO, BytesIO

# ======== constant values =========
default_sed_filter_file = os.path.join(os.path.dirname(__file__), '../filters/wer_filter')


# ==================================

def is_py2():
    return sys.version_info.major == 2


def is_py3():
    return sys.version_info.major == 3


def ensure_str(str_or_bytes):
    if isinstance(str_or_bytes, bytes):
        return str_or_bytes.decode()
    return str_or_bytes

class Logger(object):
    """output text to log file and console"""

    def __init__(self, log_file, mod='at'):
        self.__console__ = sys.__stdout__
        mkdir(os.path.dirname(log_file))
        self.__log__ = open(log_file, mod)

    def __del__(self):
        self.__log__.close()

    def write(self, output_stream):
        self.__console__.write(output_stream)
        self.__log__.write(output_stream)
        self.flush()

    def flush(self):
        self.__console__.flush()
        self.__log__.flush()


def mkdir(path_name):
    """
    make dirs if not exist

    Args:
        path_name: dir path

    Returns:
        path
    """
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    return path_name


def path_file_name(path, rm_suffix=False):
    """
    extract the file name from a path

    For example:
        path = '/Users/lls/data/a.txt'
        rm_suffix = True
        return 'a'

        path = '/Users/lls/data/a.txt'
        rm_suffix = False
        return a.txt

    Args:
        path: str
        rm_suffix:  if True, then remove the suffix

    Returns:

    """
    s = os.path.split(path)[-1]
    if rm_suffix:
        s = os.path.splitext(s)[0]
    return s


def path_replace_suffix(path, suffix_to):
    return os.path.splitext(path)[0] + suffix_to


def check_cmd(cmd_str, help='exit'):
    a = os.system('which %s > /dev/null' % cmd_str.split()[0])
    assert a == 0, 'cannot find cmd %s, %s' % (cmd_str, help)
    return True


def run_cmd(cmd_str, input_stream=None, output_stream=None, error_stream=None):
    """
    Note: byte in, byte out
    """
    p = subprocess.Popen(cmd_str,
                         stdin=subprocess.PIPE,
                         stdout=None if output_stream is None else subprocess.PIPE,
                         stderr=None if error_stream is None else subprocess.PIPE,
                         shell=True)

    # run input
    if input_stream is not None:
        with SaveOpen(input_stream, 'rb') as fin:
            out_str, err_str = p.communicate(fin.read())
    else:
        out_str, err_str = p.communicate()

    # get output
    if output_stream is not None:
        with SaveOpen(output_stream, 'wb') as fout:
            fout.write(out_str)

    # get error
    if error_stream is not None:
        with SaveOpen(error_stream, 'wb') as ferr:
            ferr.write(err_str)

    return output_stream, error_stream


def run_sox_silence(cmd_str, input_stream=None, output_stream=None, error_stream=None):
    """
    to remove the warning in sox such as:
        "sox WARN wav: Length in output .wav header will be wrong since can't seek to fix it"
    """
    return run_cmd(cmd_str.replace('sox', 'sox -V0'),
                   input_stream=input_stream,
                   output_stream=output_stream,
                   error_stream=error_stream)


def filter_text(input_stream, output_stream, sed_filter_file=None):
    processed_input_str = ""
    with SaveOpen(input_stream, 'rt') as f:
        for line in f:
            processed_input_str += '  ' + '  '.join(line.split()) + '  ' + '\n'

    sed_filter_file = sed_filter_file if sed_filter_file else default_sed_filter_file

    cmd_str = 'sed -f %s | awk \'{$1=$1; print}\'' % sed_filter_file

    return run_cmd(cmd_str, BytesIO(processed_input_str.encode()), output_stream)[0]


def audio_duration(path):
    _, s = subprocess.Popen('sox {} -n stat'.format(path), stderr=subprocess.PIPE, shell=True).communicate()

    s = ensure_str(s)

    for line in s.strip().split('\n'):
        if line.lower().startswith('length (seconds):'):
            return float(line.split()[-1])
    raise TypeError('sox ERROR: \n{}'.format(s))


def generate_bpe(input_stream, out_bpe_stream, out_vocab_stream, wp_size, to_lower=False):
    """
    run tools https://github.com/albertz/subword-nmt.git to train bpe and vocab
    please run pip install subword-nmt to install the tools

    Args:
        input_stream: input word text
        out_bpe_stream: output bpe codes
        out_vocab_stream: output vocab
        wp_size: wp size (operation size)
        to_lower: boolean

    Returns:
        None
    """
    check_cmd('subword-nmt', help='please run "pip install subword-nmt" to install subword-nmt')

    if to_lower:
        base_cmd = "tr '[:upper:]' '[:lower:]' | "
    else:
        base_cmd = ''

    # learn bpe
    run_cmd(base_cmd + 'subword-nmt learn-bpe -s {} '.format(wp_size),
            input_stream=input_stream, output_stream=out_bpe_stream)

    # word to wp
    wp_stream = generate_wp(
        input_stream=input_stream,
        output_stream=BytesIO(),
        bpe_codes=out_bpe_stream,
        to_lower=to_lower
    )

    run_cmd(base_cmd + 'subword-nmt get-vocab',
            input_stream=wp_stream,
            output_stream=out_vocab_stream)


def generate_wp(input_stream, output_stream, bpe_codes, vocabulary=None, to_lower=False):
    """
    run tools https://github.com/albertz/subword-nmt.git to generate word-piece from word
    please run pip install subword-nmt to install the tools

    Args:
        input_stream: input stream
        output_stream: output stream
        bpe_codes: bpe code file
        vocabulary: vocab file
        to_lower: if True, then transform the characters to lower

    Returns:
        output_stream
    """
    check_cmd('subword-nmt', help='please run "pip install subword-nmt" to install subword-nmt')

    cmd_str = 'subword-nmt apply-bpe'
    cmd_str += ' -c ' + bpe_codes
    if vocabulary is not None:
        cmd_str += ' --vocabulary ' + vocabulary
    if to_lower:
        cmd_str = "tr '[:upper:]' '[:lower:]' | " + cmd_str

    return run_cmd(cmd_str, input_stream, output_stream)[0]


def generate_wp_from_list(input_sent_list, bpe_codes, vocabulary=None, to_lower=False):
    """
    input a list of sentences, generate word-pieces and return a list of wp-sentences
    """
    input_stream = StringIO('\n'.join(input_sent_list) + '\n')
    output_str = stream2str(
        generate_wp(input_stream=input_stream,
                    output_stream=BytesIO(),
                    bpe_codes=bpe_codes,
                    vocabulary=vocabulary,
                    to_lower=to_lower)
    )
    out_list = output_str.split('\n')[0: -1]
    assert len(out_list) == len(input_sent_list)
    return out_list


def read_txt_to_wp_dict(input_stream, bpe_codes, vocabulary=None, wp_vocab_path=None):
    """
    read a sentence scp file and transform to wp or wp-ids.
    
    Args:
        input_stream: a scp text file
        bpe_codes: bpe codes
        vocabulary: bpe vocabulary
        wp_vocab_path: a vocabulary to convert wp-str to wp-ids, if None, then return the wp-str

    Returns:
        a dict mapping keys to wp-strings/wp-ids
    """
    if wp_vocab_path is not None:
        w2id = load_vocab(wp_vocab_path)[0]

    def sent_to_wp_ids(s):
        s = generate_wp_from_list([s], bpe_codes, vocabulary)[0]
        if wp_vocab_path is not None:
            ids = [w2id[w] for w in s.split()]
        else:
            ids = s.split()
        return ids

    return read_txt_to_dict(input_stream, value_fun=sent_to_wp_ids)


def convert_speech(input_stream, output_stream,
                   from_type,
                   to_type='wav',
                   to_bit=16,
                   to_sample_rate=16000):
    cmd_str = 'sox -t {} - -t {} -c1 -b {} -r {} -'.format(from_type, to_type, to_bit, to_sample_rate)
    run_sox_silence(cmd_str,
                    input_stream=input_stream,
                    output_stream=output_stream)
    return output_stream


def pad_seqs_to_matrix(seq_list, pad_val=0, dtype=None):
    if dtype is None:
        dtype = np.array(seq_list[0]).dtype

    seq_len = [len(s) for s in seq_list]
    max_len = max(seq_len)
    pad_list = [
        np.pad(a, [(0, max_len - n)] + [(0, 0)] * (np.ndim(a) - 1), 'constant', constant_values=pad_val)
        for a, n in zip(seq_list, seq_len)
    ]
    return np.stack(pad_list, axis=0).astype(dtype), np.array(seq_len).astype('int32')


def split_pad_seqs_to_list(pad_seqs, pad_lengths):
    res = []
    for x, n in zip(pad_seqs, pad_lengths):
        res.append(x[0: n])
    return res


def wps_to_words(wps):
    """
    input wp sequence and output word sequence
    For example: this is a g@@ oo@@ d day -> this is a good day
    """
    s = ''
    for wp in wps:
        if wp.endswith('@@'):
            s += wp[0:-2]
        else:
            s += wp + ' '
    return s.split()


def fast_shuffle(x):
    idx = np.random.permutation(len(x))
    return [x[i] for i in idx]


def sox_merge_wav_line(wav_line_list):
    a = []
    for wav_line in wav_line_list:
        wav_line = wav_line.strip()
        if wav_line.endswith('|'):
            a.append('-t wav "|{}"'.format(wav_line.rstrip('|')))
        else:
            a.append('-t wav {}'.format(wav_line))

    s = 'sox ' + ' '.join(a) + ' -t wav - |'
    return s


def is_str(s):
    try:
        return isinstance(s, (str, unicode))
    except:
        return isinstance(s, str)
