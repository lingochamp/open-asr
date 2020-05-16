import os
import sys
import numpy as np
import collections
import subprocess
import h5py
from collections import OrderedDict
from absl import logging
import json
import glob


try:
    # python 2
    import StringIO as _io
    # class StringIO(_io.StringIO):
    #     pass
    # class BytesIO(_io.StringIO):
    #     pass
    StringIO = _io.StringIO
    BytesIO = _io.StringIO

    str_types = (str, unicode)
except:
    # python 3
    from io import StringIO
    from io import BytesIO
    str_types = str


class SaveOpen(object):
    """support file name (str), file_handler, or StringIO """

    def __init__(self, str_or_stream, open_fmt='rt'):
        self.str_or_stream = str_or_stream
        self.open_fmt = open_fmt
        self.fp = None

    def is_text(self):
        return not self.open_fmt.endswith('b')

    def is_read(self):
        return self.open_fmt.find('r') != -1

    def get(self):
        if self.str_or_stream is None:
            self.fp = None
            return

        if isinstance(self.str_or_stream, str_types):
            self.fp = open(self.str_or_stream, self.open_fmt)
        elif isinstance(self.str_or_stream, bytes):
            self.fp = open(self.str_or_stream.decode(), self.open_fmt)
        elif StringIO != BytesIO and isinstance(self.str_or_stream, StringIO) and not self.is_text():
            if not self.is_read():
                raise TypeError('write to StringIO using fmt {}'.format(self.open_fmt))
            # switch stringio to bytesio
            self.str_or_stream.seek(0)
            s = self.str_or_stream.read()
            self.fp = BytesIO(s)
        elif StringIO != BytesIO and isinstance(self.str_or_stream, BytesIO) and self.is_text():
            if not self.is_read():
                raise TypeError('write to BytesIO using fmt {}'.format(self.open_fmt))
            # switch bytesio to stringio
            self.str_or_stream.seek(0)
            self.fp = StringIO(self.str_or_stream.read().decode(errors='ignore'))
        else:
            self.fp = self.str_or_stream

        # reset the file stream
        if self.open_fmt.find('a') == -1:
            self.fp.seek(0)
        else:
            self.fp.seek(0, 2)
        return self.fp

    def __enter__(self):
        return self.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp is None:
            return

        if isinstance(self.str_or_stream, str):
            self.fp.close()
        else:
            self.fp.seek(0)


def stream2str(input_stream):
    with SaveOpen(input_stream, 'rt') as f:
        s = f.read()
    return s


def stream2bytes(input_stream):
    with SaveOpen(input_stream, 'rb') as f:
        s = f.read()
    return s



def file_lines(input_stream):
    i = 0
    with SaveOpen(input_stream, 'rt') as f:
        for _ in f:
            i += 1
        f.seek(0)
    return i


def read_txt_to_dict(input_stream, filter_fun=None, value_fun=str):
    """
    read text to dict
    For example:
        input: 'key1 a a b b\n'
               'key2 a b c d\n'

        return: {'key1', 'a a b b',
                 'key2', 'a b c d'}

    Args:
        input_stream: a file path
        filter_fun: a function return bool.
        value_fun:  a function used to process the line

    Returns:
        an OrderDict()
    """
    data = collections.OrderedDict()
    skipped_lines = 0

    with SaveOpen(input_stream, 'rt') as f:
        for line in f:
            a = line.strip().split(None, 1)
            label = a[0]
            if len(a) >= 2:
                info = a[1]
            else:
                info = ''

            if filter_fun is not None and not filter_fun(info):
                skipped_lines += 1
                continue
            data[label] = value_fun(info)

        logging.debug('from {} load {} lines, skip {} ({:.2f}%) lines'.format(
            input_stream, len(data), skipped_lines,
            100.0 * skipped_lines / max(1, skipped_lines + len(data))
        ))
    return data


def read_lines_to_dict(input_stream, filter_fun=None, value_fun=str, key_fun=str):
    """
    read each lines and call key_fun to get the key, and call value_fun to extract the value

    All the inputs of filter_fun/value_fun/key_fun are the line string

    Returns:
        an OrderDict
    """
    data = collections.OrderedDict()
    skipped_lines = 0

    with SaveOpen(input_stream, 'rt') as f:
        for line in f:
            line = line.strip()

            if filter_fun is not None and not filter_fun(line):
                skipped_lines += 1
                continue

            data[key_fun(line)] = value_fun(line)

        logging.debug('from {} load {} lines, skip {} ({:.2f}%) lines'.format(
            input_stream, len(data), skipped_lines,
            100.0 * skipped_lines / (skipped_lines + len(data))
        ))
    return data


def write_dict_to_txt(data_dict, output_stream, format_fun=str):
    with SaveOpen(output_stream, 'wt') as fout:
        for key, val in data_dict.items():
            fout.write('{} {}\n'.format(key, format_fun(val)))
    return output_stream


def read_score(score_stream):
    return np.array(list(read_txt_to_dict(score_stream, value_fun=float).values()))


def write_score(score_array, write_stream, keys=None):
    with SaveOpen(write_stream, 'wt') as fout:
        for i in range(len(score_array)):
            if keys is None:
                key = 'utt%d' % i
            else:
                key = keys[i]
            fout.write('{} {}\n'.format(key, score_array[i]))
    return write_stream


def read_keys(input_stream):
    keys = []
    with SaveOpen(input_stream) as f:
        for line in f:
            keys.append(line.split()[0])
    return keys


def dict_to_nbest(src_dict, label_sep='-'):
    d = collections.OrderedDict()
    for label, val in src_dict.items():
        root_key = label_sep.join(label.split(label_sep)[0:-1])
        d.setdefault(root_key, [])
        d[root_key].append(val)
    return d


def read_txt_to_nbest_dict(input_stream, filter_fun=None, value_fun=str, label_sep='-'):
    """
    read text to dict
    For example:
        input: 'key1-1 a\n'
               'key1-2 b\n'
               'key2-1 c\n'
               'key2-2 d\n'

        return: {'key1', ['a', 'b'],
                 'key2', ['c', 'd'],}

    Args:
        input_stream: a file path
        filter_fun: a function return bool.
        value_fun:  a function used to process the line
        label_sep: the separator for key and hypos_id

    Returns:
        an OrderDict()
    """

    data = collections.OrderedDict()
    skipped_lines = 0

    with SaveOpen(input_stream, 'rt') as f:
        for line in f:
            a = line.strip().split(None, 1)
            label = a[0]
            if len(a) >= 2:
                info = a[1]
            else:
                info = ''

            if filter_fun is not None and not filter_fun(info):
                skipped_lines += 1
                continue
            root_key = label_sep.join(label.split(label_sep)[0:-1])
            data.setdefault(root_key, [])
            data[root_key].append(value_fun(info))

        logging.debug('from {} load {} lines, skip {} ({:.2f}%) lines'.format(
            input_stream, len(data), skipped_lines,
            100.0 * skipped_lines / (skipped_lines + len(data))
        ))
    return data


def write_nbest_dict_to_txt(data_dict, output_stream, format_fun=str, label_sep='-'):
    with SaveOpen(output_stream, 'wt') as fout:
        for key, vals in data_dict.items():
            for i, val in enumerate(vals):
                fout.write('{} {}\n'.format(key + label_sep + str(i), format_fun(val)))

    return output_stream


def nbest_to_best(nbest_stream, best_stream, score_stream=None):
    d = read_txt_to_nbest_dict(nbest_stream)

    if score_stream is None:
        # chose the first
        logging.info('input score is None, select the first')
        best_d = collections.OrderedDict(
            [(key, vals[0]) for key, vals in d.items()]
        )
    else:
        # chose the best
        score_dict = read_txt_to_nbest_dict(score_stream, value_fun=float)
        best_d = collections.OrderedDict(
            [(key, vals[np.argmin(score_dict[key])]) for key, vals in d.items()]
        )
    write_dict_to_txt(best_d, best_stream)


def best_refer_iter(best_stream, refer_stream):
    best_dict = read_txt_to_dict(best_stream)
    refer_dict = read_txt_to_dict(refer_stream)

    for key in refer_dict.keys():
        yield best_dict[key], refer_dict[key], key


def nbest_refer_iter(nbest_stream, refer_stream, mod, score_stream=None):
    """
    read nbest and refer file to iter, which can be used to compute wer
    Args:
        nbest_stream: nbest file
        refer_stream: refer file
        mod: 3 values:
            'nbest': all the nbest are saved, and used to compute oracle WER.
            'best' / 'min_score':  select the best hypothesis (with minimum score) based on the score.
            'max_score': select the hypothesis with maximum score based on the score.
            'first': select the first hypothesis.
        score_stream:
            score stream or array or list, used when mod=='best' 'min_score' 'max_score'
    Returns:

    """
    nbest_dict_flat = read_txt_to_dict(nbest_stream)
    nbest_dict = dict_to_nbest(nbest_dict_flat)
    refer_dict = read_txt_to_dict(refer_stream)

    if mod == 'nbest':
        for key in refer_dict.keys():
            yield nbest_dict[key], refer_dict[key], key
    elif mod == 'min_score' or mod == 'max_score' or mod == 'best':
        assert score_stream is not None, 'need score stream'

        if mod == 'min_score' or mod == 'best':
            select_fun = np.argmin
        else:
            select_fun = np.argmax

        if isinstance(score_stream, (np.ndarray, list)):
            score_dict = dict_to_nbest(
                collections.OrderedDict([(key, s) for key, s in zip(nbest_dict_flat.keys(), score_stream)])
            )
        else:
            score_dict = read_txt_to_nbest_dict(score_stream, value_fun=float)

        for key in refer_dict.keys():
            opt_i = select_fun(score_dict[key])
            yield nbest_dict[key][opt_i], refer_dict[key], key
    elif mod == 'first':
        for key in refer_dict.keys():
            yield nbest_dict[key][0], refer_dict[key], key
    else:
        raise TypeError('unknown mod = {}'.format(mod))


class H5Cacher(object):
    """
    cash the h5 file handler,
    define an OrderedDict() to hash the h5 handler,
    if the hashed handler number is larger than the maximum number,
    then the unused handler is deleted.
    """

    def __init__(self, max_num=100):
        self.handler_dict = OrderedDict()
        self.max_num = max_num

    def open_or_get_cached(self, fname, fmt=None):
        if fname not in self.handler_dict:
            self.handler_dict[fname] = h5py.File(fname, fmt)
            if len(self.handler_dict) > self.max_num:
                pop_fname, pop_hd = self.handler_dict.popitem(False)  # pop the earliest h5 handler
                pop_hd.close()
        else:
            # move the handler to the latest
            hd = self.handler_dict.pop(fname)
            self.handler_dict[fname] = hd

        return self.handler_dict[fname]

    def get(self, fname, key):
        return self.open_or_get_cached(fname)[key]


def dataset_info(dataset_dir, dur_file='utt2dur', text_file='text'):
    """
    given a kaldi based dataset dir, get the data informations

    Args:
        dataset_dir: dataset dir
        dur_file: the file name of utt2dur
        text_file: the file name of text

    Returns:

    """

    def text_stat(file_or_str_list):
        if isinstance(file_or_str_list, str):
            a = open(file_or_str_list).readlines()
        elif isinstance(file_or_str_list, list):
            a = file_or_str_list
        else:
            a = file_or_str_list.readlines()

        wdict = dict()
        sdict = dict()
        for s in a:
            for w in s.split():
                wdict.setdefault(w, 0)
                wdict[w] += 1
            s = ' '.join(s.split())
            sdict.setdefault(s, 0)
            sdict[s] += 1
        return wdict, sdict

    if not os.path.isdir(dataset_dir):
        raise TypeError('This is not a dataset dir: {}'.format(dataset_dir))

    res = OrderedDict()

    data_dur = read_txt_to_dict(os.path.join(dataset_dir, dur_file))
    durs = [float(x) for x in data_dur.values()]
    res['dur_total'] = '{:.2f} hours'.format(sum(durs) / 3600)
    res['dur_min'] = '{:.2f} s'.format(min(durs))
    res['dur_max'] = '{:.2f} s'.format(max(durs))

    data_text = read_txt_to_dict(os.path.join(dataset_dir, text_file))
    wdict, sdict = text_stat(list(data_text.values()))
    res['word_count'] = sum(wdict.values())
    res['word_unique'] = len(wdict)
    res['sent_count'] = sum(sdict.values())
    res['sent_unique'] = len(sdict)
    res['sent_avg_repeat'] = 1.0 * res['sent_count'] / res['sent_unique']
    repeats = sorted(sdict.values())
    res['sent_repeats'] = str([repeats[0],
                               repeats[len(repeats) // 4],
                               repeats[len(repeats) // 2],
                               repeats[-len(repeats) // 4],
                               repeats[-1]])
    lens = [len(x) for x in data_text.values()]
    res['sent_min_len'] = min(lens)
    res['sent_max_len'] = max(lens)

    logging.info('dataset : {}'.format(dataset_dir))
    logging.info(json.dumps(res, indent=2))
    return res


def load_vocab(input_stream, word_column=0, id_column=None, inserts=None):
    """
    load a vocabulary form file
    choice the word_column column as the word string;
    choice the id_column column as the word-id.
    If id_column is None (default), then using the line number as the word-id (from 0)

    Args:
        input_stream: a file stream
        word_column: integer
        id_column: integer or None
        inserts: a list of two value tuples.
            The first value is id, the second value is word
            For example:
                inserts=[(0, '<eps>'), (-1, '<unk>'), ('last', '<tail>')]
                this will add <eps> to the head of the vocabulary,
                and add <unk> to the last but one of the vocabulary (NOTE: not the last).
                using flag 'last' to append to the vocab tail

    Returns:
        a tuple including two values
        - wdict: word-to-id dict
        - wlist: id-to-word list
    """
    wdict = dict()
    with SaveOpen(input_stream) as f:
        num_lines = 0
        for line in f:
            a = line.split()
            w = a[word_column]
            i = int(a[id_column]) if id_column is not None else num_lines
            wdict[w] = i
            num_lines += 1

    wlist = list(map(lambda x: x[0], sorted(wdict.items(), key=lambda x: x[1])))
    num_word = len(wlist)
    max_id = max(wdict.values())
    assert num_word == max_id + 1, 'the vocab is not continuous: max-id={} word-num={}'.format(max_id, num_word)

    if inserts:
        for i, w in inserts:
            if i == 'last':
                wlist.append(w)
            else:
                wlist.insert(i, w)

        # recreate w2id
        wdict = dict(map(lambda x: (x[1], x[0]), enumerate(wlist)))

    return wdict, wlist


class StreamDict(object):
    """
    a class can used to read large dict list file, such as text, wav
    """

    def __init__(self, file_path, filter_fun=None, value_fun=str, fetch_lines=1000):
        self.fp = SaveOpen(file_path, 'rt').get()
        self.filter_fun = filter_fun
        self.value_fun = value_fun

        self.data_dict = dict()
        self.fetch_lines = fetch_lines
        self.all_loaded = False

    def get_next_lines(self):
        lines = []
        for s in self.fp:
            s = s.strip('\n')
            lines.append(s)
            if len(lines) >= self.fetch_lines:
                break
        return lines

    def get_next_dict(self):
        lines = self.get_next_lines()
        if lines:
            new_dict = read_txt_to_dict(StringIO('\n'.join(lines)),
                                        filter_fun=self.filter_fun,
                                        value_fun=self.value_fun)
            self.data_dict.update(new_dict)
            self.all_loaded = False
        else:
            self.all_loaded = True
        return self.data_dict

    def get(self, key):

        while key not in self.data_dict and not self.all_loaded:
            # update data_dict
            self.get_next_dict()

        return self.data_dict[key]

    def __getitem__(self, item):
        return self.get(item)


def dict_select(d, keys):
    return OrderedDict([(k, d[k]) for k in keys])


def read_dataset_to_dicts(data_dir):
    return (
        read_txt_to_dict(data_dir + '/wav.scp'),
        read_txt_to_dict(data_dir + '/text'),
        read_txt_to_dict(data_dir + '/utt2dur', value_fun=float)
    )


def write_dicts_to_dataset(out_dir, wav_dict=None, txt_dict=None, dur_dict=None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if wav_dict:
        write_dict_to_txt(wav_dict, out_dir + '/wav.scp')
    if txt_dict:
        write_dict_to_txt(txt_dict, out_dir + '/text')
    if dur_dict:
        write_dict_to_txt(dur_dict, out_dir + '/utt2dur')
