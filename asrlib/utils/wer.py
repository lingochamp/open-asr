import os
import sys
import numpy as np
from absl import logging
import time

from asrlib.utils import base, reader
from asrlib.utils.base import BytesIO, StringIO


def txt_score(hypos, refer, special_word=None):
    """

    compute the err number

    For example:

        refer = A B C
        hypos = X B V C

        after alignment

        refer = ~A B ^  C
        hypos = ~X B ^V C
        err number = 2

        where ~ denotes replacement error, ^ denote insertion error, * denotes deletion error.

    if set the special_word (not None), then the special word in reference can match any words in hypothesis.
    For example:
        refer = 'A <?> C'
        hypos = 'A B C D C'

        after aligment:

        refer =  A   <?> C
        hypos =  A B C D C

        where <?> matches 'B C D'. error number = 0

    Usage:
        ```
        refer = 'A <?> C'
        hypos = 'A B C D C'

        res = wer.wer(refer, hypos, '<?>')
        print('err={}'.format(res['err']))
        print('ins={ins} del={del} rep={rep}'.format(**res))
        print('refer = {}'.format(' '.join(res['refer'])))
        print('hypos = {}'.format(' '.join(res['hypos'])))
        ```

    Args:
        hypos: a string or a list of words
        refer: a string or a list of hypos
        special_word: this word in reference can match any words

    Returns:
        a result dict, including:
        res['word']: word number
        res['err']:  error number
        res['del']:  deletion number
        res['ins']:  insertion number
        res['rep']:  replacement number
        res['hypos']: a list of words, hypothesis after alignment
        res['refer']: a list of words, reference after alignment
    """

    res = {'word': 0, 'err': 0, 'none': 0, 'del': 0, 'ins': 0, 'rep': 0, 'hypos': [], 'refer': []}

    refer_words = refer if isinstance(refer, list) else refer.split()
    hypos_words = hypos if isinstance(hypos, list) else hypos.split()

    hypos_words.insert(0, '<s>')
    hypos_words.append('</s>')
    refer_words.insert(0, '<s>')
    refer_words.append('</s>')

    hypos_len = len(hypos_words)
    refer_len = len(refer_words)

    if hypos_len == 0 or refer_len == 0:
        return res

    go_nexts = [[0, 1], [1, 1], [1, 0]]
    score_table = [([['none', 10000, [-1, -1], '', '']] * refer_len) for hypos_cur in range(hypos_len)]
    score_table[0][0] = ['none', 0, [-1, -1], '', '']  # [error-type, note distance, best previous]

    for hypos_cur in range(hypos_len - 1):
        for refer_cur in range(refer_len):

            for go_nxt in go_nexts:
                hypos_next = hypos_cur + go_nxt[0]
                refer_next = refer_cur + go_nxt[1]
                if hypos_next >= hypos_len or refer_next >= refer_len:
                    continue

                next_score = score_table[hypos_cur][refer_cur][1]
                next_state = 'none'
                next_hypos = ''
                next_refer = ''

                if go_nxt == [0, 1]:
                    if special_word is not None and refer_words[refer_next] == special_word:
                        next_state = 'none'
                        next_score += 0
                        next_hypos = ' ' * len(refer_words[refer_next])
                        next_refer = refer_words[refer_next]

                    else:
                        next_state = 'del'
                        next_score += 1
                        next_hypos = '*' + ' ' * len(refer_words[refer_next])
                        next_refer = '*' + refer_words[refer_next]

                elif go_nxt == [1, 0]:
                    next_state = 'ins'
                    next_score += 1
                    next_hypos = '^' + hypos_words[hypos_next]
                    next_refer = '^' + ' ' * len(hypos_words[hypos_next])

                else:
                    if special_word is not None and refer_words[refer_next] == special_word:
                        for ii in range(hypos_cur + 1, hypos_len - 1):
                            next_score += 0  # can match any words, without penalty
                            next_state = 'none'
                            next_refer = special_word
                            next_hypos = ' '.join(hypos_words[hypos_cur + 1:ii + 1])

                            if next_score < score_table[ii][refer_next][1]:
                                score_table[ii][refer_next] = [next_state, next_score, [hypos_cur, refer_cur],
                                                               next_hypos, next_refer]

                        # avoid add too many times
                        next_score = 10000

                    else:
                        next_hypos = hypos_words[hypos_next]
                        next_refer = refer_words[refer_next]
                        if hypos_words[hypos_next] != refer_words[refer_next]:
                            next_state = 'rep'
                            next_score += 1
                            next_hypos = '~' + next_hypos
                            next_refer = '~' + next_refer

                if next_score < score_table[hypos_next][refer_next][1]:
                    score_table[hypos_next][refer_next] = [next_state, next_score, [hypos_cur, refer_cur], next_hypos,
                                                           next_refer]

    res['err'] = score_table[hypos_len - 1][refer_len - 1][1]
    res['word'] = refer_len - 2
    hypos_cur = hypos_len - 1
    refer_cur = refer_len - 1
    refer_fmt_words = []
    hypos_fmt_words = []
    while hypos_cur >= 0 and refer_cur >= 0:
        res[score_table[hypos_cur][refer_cur][0]] += 1  # add the del/rep/ins error number
        hypos_fmt_words.append(score_table[hypos_cur][refer_cur][3])
        refer_fmt_words.append(score_table[hypos_cur][refer_cur][4])
        [hypos_cur, refer_cur] = score_table[hypos_cur][refer_cur][2]

    refer_fmt_words.reverse()
    hypos_fmt_words.reverse()

    # format the hypos and refer
    assert len(refer_fmt_words) == len(hypos_fmt_words)
    for hypos_cur in range(len(refer_fmt_words)):
        w = max(len(refer_fmt_words[hypos_cur]), len(hypos_fmt_words[hypos_cur]))
        fmt = '{:>%d}' % w
        refer_fmt_words[hypos_cur] = fmt.format(refer_fmt_words[hypos_cur])
        hypos_fmt_words[hypos_cur] = fmt.format(hypos_fmt_words[hypos_cur])

    res['refer'] = refer_fmt_words[1:-1]
    res['hypos'] = hypos_fmt_words[1:-1]

    return res


def write_wer_res(output_stream, utt_i, key, res, total_err, total_word, res_list=None, time_begin=None):
    output_stream.write('label: {}\n'.format(key))
    output_stream.write('utt: {}\n'.format(utt_i))
    output_stream.write('err: {} / {} {} / {} avg-wer={:.3f}\n'.format(
        res['err'], res['word'], total_err, total_word, 1.0 * total_err / total_word if total_word else 0.0))
    if res_list is not None:
        output_stream.write('errs: ')
        for res in res_list:
            output_stream.write('{}/{}/{}/{}/{} '.format(res['ins'], res['del'], res['rep'], res['err'], res['word']))
        output_stream.write('\n')
    if time_begin is not None:
        t = (time.time() - time_begin) / 60.0
        output_stream.write('time: total {:.4f} min, {:.4f} min per utterance\n'.format(t, t/(utt_i + 1)))
    output_stream.write('refer: ' + ''.join([i + ' ' for i in res['refer']]) + '\n')
    output_stream.write('hypos: ' + ''.join([i + ' ' for i in res['hypos']]) + '\n')
    output_stream.write('==========\n')
    output_stream.flush()


def compute_wer(hypos_refer_iter, special_word=None, output_stream=None):
    """
    input a set of (hypothesis, reference), compute the final wer

    Args:
        hypos_refer_iter: a iter return a tuple of (hypos , refer) or a tuple of (hypos, refer, key)
            the refer and hypos can be either word list or a sentence string
        special_word: special word
        output_stream: output stream

    Returns:
        (total_err, total_word, WER)
    """
    total_word = 0
    total_err = 0
    total_ins_err = 0
    total_del_err = 0
    total_sub_err = 0
    time_begin = time.time()

    for utt_i, iter_tuple in enumerate(hypos_refer_iter):
        if len(iter_tuple) == 2:
            hypos, refer = iter_tuple[0: 2]
            key = 'unknown'
        elif len(iter_tuple) == 3:
            hypos, refer, key = iter_tuple
        else:
            raise TypeError('can not parser the values in iter: {}'.format(iter_tuple))

        res = txt_score(hypos, refer, special_word=special_word)
        total_err += res['err']
        total_ins_err += res['ins']
        total_del_err += res['del']
        total_sub_err += res['rep']
        total_word += res['word']

        if output_stream is not None:
            write_wer_res(output_stream, utt_i, key, res, total_err, total_word, res_list=[res], time_begin=time_begin)

    if output_stream is not None:
        output_stream.write('[Finished]\n')
        output_stream.write(
            'total_err = {} | ins={} del={} sub={}\n'.format(total_err, total_ins_err, total_del_err, total_sub_err))
        output_stream.write('total_word = {}\n'.format(total_word))
        output_stream.write('wer = {:.6f}\n'.format(100.0 * total_err / total_word))

    return total_err, total_word, 100.0 * total_err / total_word


def compute_oracle_map_func(input_tuple):
    hypos_list, refer, key, special_word = input_tuple

    res_list = []
    opt_i = None
    for i, hypos in enumerate(hypos_list):
        res = txt_score(hypos, refer, special_word=special_word)
        if opt_i is None or res['err'] < res_list[opt_i]['err']:
            opt_i = i
        res_list.append(res)

    return key, res_list, opt_i


def compute_oracle_wer(nbest_refer_iter, special_word=None, output_stream=None, processes=4, total=None):
    """
    compute the oracle WER
    Args:
        nbest_refer_iter: a iter return a tuple of (a list of hypos , refer) or a tuple of (a list of hypos, refer, key)
        special_word: special word
        output_stream: output stream
        processes: multi-process number

    Returns:
        (total_err, total_word, WER)
    """
    import tqdm
    from multiprocessing import Pool

    def map_iter():
        for iter_tuple in nbest_refer_iter:
            if len(iter_tuple) == 2:
                hypos, refer = iter_tuple[0: 2]
                key = 'unknown'
            elif len(iter_tuple) == 3:
                hypos, refer, key = iter_tuple
            else:
                raise TypeError('can not parser the values in iter: {}'.format(iter_tuple))

            yield hypos, refer, key, special_word

    total_err = 0
    total_word = 0
    utt_num = 0
    pool = Pool(processes=processes)
    for key, res_list, opt_i in tqdm.tqdm(pool.imap(compute_oracle_map_func, map_iter()),
                                          desc='compute oracle WER',
                                          total=total):

        opt_res = res_list[opt_i]
        total_err += opt_res['err']
        total_word += opt_res['word']

        if output_stream is not None:
            write_wer_res(output_stream, utt_num, key, opt_res, total_err, total_word,
                          res_list=res_list)

        utt_num += 1

    return total_err, total_word, 100.0 * total_err / total_word


class Nbest(object):
    def __init__(self, nbest_file, refer_file,
                 acscore_file=None, lmscore_file=None, gfscore_file=None,
                 sed_filter_file=base.default_sed_filter_file,
                 ):
        self.acscore = reader.read_score(acscore_file) if acscore_file is not None else None
        self.lmscore = reader.read_score(lmscore_file) if lmscore_file is not None else None
        self.gfscore = reader.read_score(gfscore_file) if gfscore_file is not None else None

        self.ac_weight = 1.0
        self.lm_weight = 1.0
        self.special_word = '<?>'
        self.sed_filter_file = sed_filter_file

        if self.sed_filter_file is None:
            self.nbest_stream = reader.SaveOpen(nbest_file).get()
            self.refer_stream = reader.SaveOpen(refer_file).get()
        else:
            self.nbest_stream = base.filter_text(nbest_file, BytesIO(), self.sed_filter_file)
            self.refer_stream = base.filter_text(refer_file, BytesIO(), self.sed_filter_file)

        self.total_utt = reader.file_lines(self.refer_stream)

        self.wer_log = None
        self.oracle_log = None

        logging.info('total utt = %d' % self.total_utt)

    def _reset_stream(self):
        self.nbest_stream.seek(0)
        self.refer_stream.seek(0)

    def create_iter(self):
        # reset the stream
        self._reset_stream()

        if self.acscore is None:
            # select the first of nbest
            return reader.nbest_refer_iter(self.nbest_stream, self.refer_stream, mod='first')
        else:
            if self.lmscore is None:
                self.lmscore = np.zeros_like(self.acscore)
            if self.gfscore is None:
                self.gfscore = np.zeros_like(self.acscore)

            score = self.ac_weight * (self.acscore + self.gfscore) + self.lm_weight * self.lmscore

            return reader.nbest_refer_iter(self.nbest_stream, self.refer_stream, mod='best', score_stream=score)

    def wer(self):
        """
        compute WER using current config

        Returns:
            wer
        """
        self.wer_log = StringIO()
        _, _, res_wer = compute_wer(self.create_iter(), self.special_word, output_stream=self.wer_log)
        self.wer_log.seek(0)
        return res_wer

    def oracle_wer(self):
        """
        compute oracle wer
        """
        self._reset_stream()

        self.oracle_log = StringIO()
        _, _, res_wer = compute_oracle_wer(
            reader.nbest_refer_iter(self.nbest_stream, self.refer_stream, mod='nbest'),
            special_word=self.special_word,
            output_stream=self.oracle_log,
            total=self.total_utt
        )
        self.oracle_log.seek(0)
        return res_wer

    def tune_wer(self, lmweights=np.linspace(0, 1.0, 11), acweights=[1.0]):
        logging.info('tune wer, lmw={}, acw={}'.format(lmweights, acweights))

        opt_acw = 0
        opt_lmw = 0
        opt_wer = 100.0
        opt_log = None

        for acw in acweights:
            for lmw in lmweights:
                self.ac_weight = acw
                self.lm_weight = lmw
                cur_wer = self.wer()
                cur_log = self.wer_log
                logging.info('acw={} lmw={} wer={}'.format(acw, lmw, cur_wer))
                if cur_wer < opt_wer:
                    opt_wer = cur_wer
                    opt_lmw = lmw
                    opt_acw = acw
                    opt_log = cur_log

        logging.info('best: acw={} lmw={} wer={}'.format(opt_acw, opt_lmw, opt_wer))
        self.lm_weight = opt_lmw
        self.ac_weight = opt_acw
        self.wer_log = opt_log
        return opt_wer

    def write_log(self, write_stream):
        assert self.wer_log is not None, 'please run wer() first.'

        with reader.SaveOpen(write_stream, 'wt') as fout:
            self.wer_log.seek(0)
            fout.write(self.wer_log.read())
