import os
from asrlib.utils import base, reader, audio
from asrlib.utils.wer import compute_wer
import time

from collections import OrderedDict
import glob
import numpy as np
from absl import logging, app, flags

flags.DEFINE_string('dataset', 'testdata/dataset', 'the dataset dir')
flags.DEFINE_string('outdir', '/tmp/out_of_open_asr', 'the output dir')
flags.DEFINE_string('api', 'tencent', 'baidu / google / xunfei / tencent')
flags.DEFINE_string('path_prefix', 'testdata/dataset/', 'add to the head of audio path')
flags.DEFINE_integer('job_num', 1, '')
flags.DEFINE_integer('job_idx', 0, '')
flags.DEFINE_bool('rerun_empty', False, 'retry if the recognized text is empty')
flags.DEFINE_bool('rerun_error', True, 'retry if the recognition cause an error')
flags.DEFINE_bool('rerun_short', False, 'retry if the recognized text is too short')
flags.DEFINE_bool('rerun_all', False, 'rerun all the sentences')
FLAGS = flags.FLAGS

sed_filter_file = base.default_sed_filter_file


def process_wav_line(wav_line):
    a = wav_line.split()
    res = []
    for s in a:
        if s.endswith('.wav') or s.endswith('.mp3'):
            s = FLAGS.path_prefix + s
        res.append(s)
    return ' '.join(res)


def audio_to_wav(wav_line, tmp_name):
    wav_line = process_wav_line(wav_line)
    return audio.parse_wav_line(wav_line)


def load_all_results():
    res_dict = OrderedDict()
    for f in glob.glob(os.path.join(FLAGS.outdir, FLAGS.api, 'result*.txt')):
        print('load result from %s' % f)
        res_dict.update(reader.read_txt_to_dict(f))
    return res_dict


def should_retry(recog_text, ref_text):
    if FLAGS.rerun_empty and recog_text.strip() == '':
        print('retry as empty')
        return True
    if FLAGS.rerun_error and recog_text == '[ERROR]':
        print('retry as error')
        return True
    if FLAGS.rerun_short and len(recog_text.split()) <= len(ref_text.split()) // 2:
        print('retry as too short !')
        return True
    return False


def main(_):
    assert FLAGS.dataset is not None
    assert FLAGS.outdir is not None
    assert FLAGS.api is not None

    if FLAGS.api == 'xunfei':
        from openasr.xunfei.iat_ws_python3 import recognize_wav
    elif FLAGS.api == 'google':
        from openasr.google.speech_to_text import recognize_wav
    elif FLAGS.api == 'baidu':
        from openasr.baidu.speech_to_text import recognize_wav
    elif FLAGS.api == 'tencent':
        from openasr.tencent.speech_to_text import recognize_wav
    else:
        raise TypeError('undefined api name = {}'.format(FLAGS.api))

    base.mkdir(FLAGS.outdir)
    result_file = os.path.join(FLAGS.outdir, FLAGS.api, 'result_{}_of_{}.txt'.format(FLAGS.job_idx, FLAGS.job_num))

    wav_dict = reader.read_txt_to_dict(FLAGS.dataset + '/wav.scp')
    txt_dict = reader.read_txt_to_dict(
        base.filter_text(FLAGS.dataset + '/text', base.BytesIO(), sed_filter_file)
    )

    # get subset
    total_num = len(wav_dict)
    cur_num = int(np.ceil(1.0 * total_num / FLAGS.job_num))
    print(cur_num)
    key_list = list(wav_dict.keys())[FLAGS.job_idx * cur_num: FLAGS.job_idx * cur_num + cur_num]
    print('process {} of {} utterances'.format(len(key_list), len(wav_dict)))

    res_dict = load_all_results()
    print('load {} cached results.'.format(len(res_dict)))

    def hypos_refer_iter():
        for key in key_list:
            refer_text = txt_dict[key]
            recog_text = res_dict.get(key, None)
            if FLAGS.rerun_all:
                recog_text = None

            for retry_times in range(10):
                if recog_text is None or should_retry(recog_text, refer_text):
                    if retry_times > 0:
                        print('waiting 10 seconds before retrying ...')
                        time.sleep(10.0)  # sleep 10 seconds
                    tmp_wav = audio_to_wav(wav_dict[key], os.path.join(FLAGS.outdir, 'wav', key + '.wav'))
                    recog_text = recognize_wav(tmp_wav)
                else:
                    break

            res_dict[key] = recog_text
            reader.write_dict_to_txt(res_dict, result_file)

            hypos = recog_text.lower()
            hypos_filter = base.filter_text(base.StringIO(hypos), base.BytesIO(), sed_filter_file)
            yield reader.stream2str(hypos_filter), refer_text, key

    compute_wer(
        hypos_refer_iter(),
        special_word='<?>',
        output_stream=base.Logger(os.path.join(FLAGS.outdir, FLAGS.api,
                                               'recog_{}_of_{}.log'.format(FLAGS.job_idx, FLAGS.job_num)),
                                  'wt'))


if __name__ == '__main__':
    app.run(main)
