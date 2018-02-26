import io
import os
import re
import gzip
import json
import h5py
import numpy as np
from utils import OrderedCounter
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer

class OpenSubtitlesQADataset(Dataset):

    def __init__(self,root, split, **kwargs):

        super(OpenSubtitlesQADataset, self).__init__()

        self.root = root
        self.split = split

        self.preprocessed_file_name = 'opensubqa.' + self.split + '.hdf5'
        self.preprocessed_file_path = os.path.join(self.root, self.preprocessed_file_name)
        self.vocab_file_name = 'vocab.json'
        self.vocab_file_path = os.path.join(self.root, self.vocab_file_name)

        # set fall back params
        self.min_occ = kwargs.get('min_occ', 50)
        self.max_utterance_length = kwargs.get('max_prompt_length', 30)

        if kwargs.get('create_data', False):
            self._create_preporcessed_data(**kwargs)
        else:
            self._load_preprocessed_data()

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def __getitem__(self, idx):
        return {
            'input_sequence':         self.dataset['question'][idx].astype('int64'),
            'input_length':  self.dataset['question_length'][idx].astype('int64'),
            'reply_sequence_in':     self.dataset['answer_input'][idx].astype('int64'),
            'reply_sequence_out':    self.dataset['answer_target'][idx].astype('int64'),
            'reply_length':    self.dataset['answer_length'][idx].astype('int64')
        }

    def __len__(self):
        return len(self.dataset['question'])

    def _preprocess(self, line):
        line=line.lower()
        line=line.replace('\'s ',' is ')
        line=line.replace('\'re ',' are ')
        line=line.replace('\'m ', ' am ')
        line=line.replace('\'ve ', ' have ')
        line=line.replace('\'ll ',' will ')
        line=line.replace('n\'t ', ' not ')
        line=line.replace(' wo not',' will not')
        line=line.replace(' ca not',' can not')
        line=re.sub('[\!;-]+','',line)
        line=re.sub('\.+','.',line)
        if line.endswith(' .'):
            line=line[:-2]

        # replace separator
        assert line.count('\t') == 1, "More than one tab found."
        line=line.replace('\t', "|||")

        # delete remaining whitespace
        line=re.sub('\s+',' ',line)
        line=line.strip()
        return line

    def _create_preporcessed_data(self, **kwargs):
        raw_data_file = os.path.join(self.root, self.split + '.txt')

        assert os.path.exists(raw_data_file), "Dataset at %s not found."%raw_data_file

        if kwargs.get('create_vocab', False) or not os.path.exists(self.vocab_file_path):
            self._create_vocab(raw_data_file, **kwargs)
        else:
            self._load_vocab()

        if os.path.exists(self.preprocessed_file_name):
            os.remove(self.preprocessed_file_path)

        dataset = defaultdict(lambda: defaultdict(list))
        id = 0
        save_every = 1000
        num_lines = sum(1 for line in open(raw_data_file, 'r'))
        tokenizer = TweetTokenizer(preserve_case=False)

        with open(raw_data_file, 'r') as file:

            print("%s raw dataset found with %i lines."%(self.split.upper(), num_lines))

            for i, line in enumerate(file):

                # preprocess and split line between question and answer
                line = self._preprocess(line)
                question, answer = line.split('|||')

                # tokenize (i.e. make list of words from string)
                question_tokens = tokenizer.tokenize(question)
                answer_tokens = tokenizer.tokenize(answer)

                # map words to indicies
                question_idx = [self.w2i.get(qw, self.w2i['<unk>']) for qw in question_tokens]
                answer_idx = [self.w2i.get(aw, self.w2i['<unk>']) for aw in answer_tokens]

                # cut off
                question_idx = question_idx[:self.max_utterance_length]
                answer_idx = answer_idx[:self.max_utterance_length-2]
                answer_idx_input = [self.sos_idx] + answer_idx
                answer_idx_target = answer_idx + [self.eos_idx]

                # save length before pad
                dataset[id]['question_length'] = len(question_idx)
                dataset[id]['answer_length'] = len(answer_idx_input)

                # pad
                dataset[id]['question'] = question_idx + [self.w2i['<pad>']] * (self.max_utterance_length - dataset[id]['question_length'])
                dataset[id]['answer_input'] = answer_idx_input + [self.w2i['<pad>']] * (self.max_utterance_length - dataset[id]['answer_length'])
                dataset[id]['answer_target'] = answer_idx_target + [self.w2i['<pad>']] * (self.max_utterance_length - dataset[id]['answer_length'])
                id += 1

                if (id % save_every == 0 and id > 0) or (i+1 == num_lines):
                    self._save_to_hdf5(dataset)
                    del dataset
                    dataset = defaultdict(lambda: defaultdict(list))

                    if i > 1000000:
                        break

        self._load_preprocessed_data()

    def _load_preprocessed_data(self):
        """Load (if exists) preprocessed dataset. If not, new one will be created."""

        if not os.path.exists(self.preprocessed_file_path):
            print("Preporcessed %s Dataset Not Found. Creating New."%(self.split.upper()))
            self._create_preporcessed_data()

        else:
            self.dataset = h5py.File(self.preprocessed_file_path, 'r')
            self._load_vocab() # TODO move this outside of create data loop

        print("%s dataset with %i points loaded."%(self.split.upper(), len(self.dataset['question'])))

    def _create_vocab(self, raw_data_file, **kwargs):

        assert self.split == 'train', "Only for training data Vocablurary can be created."

        print("Creating New Vocablurary.")

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(raw_data_file) as file:

            for i, line in enumerate(file):
                line = self._preprocess(line)
                question, answer = line.split('|||')
                question = tokenizer.tokenize(question)
                question = question[:self.max_utterance_length]
                answer = tokenizer.tokenize(answer)
                answer = answer[:self.max_utterance_length-1] # sos or eos token will be added
                words = question + answer
                w2c.update(question+answer)

                if i > 1000000:
                    break

            for w, c in w2c.items():
                if c > self.min_occ:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab_file_path = os.path.join(self.root, self.vocab_file_name)
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(vocab_file_path, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

    def _load_vocab(self):

        vocab_file_path = os.path.join(self.root, self.vocab_file_name)
        with open(vocab_file_path, 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _save_to_hdf5(self, data):

        if not os.path.exists(self.preprocessed_file_path):
            # create hdf5 file
            file = h5py.File(self.preprocessed_file_path, 'w')
            file.create_dataset('question', (0, self.max_utterance_length), maxshape=(None, self.max_utterance_length), dtype='i')
            file.create_dataset('answer_input', (0, self.max_utterance_length), maxshape=(None, self.max_utterance_length), dtype='i')
            file.create_dataset('answer_target', (0, self.max_utterance_length), maxshape=(None, self.max_utterance_length), dtype='i')
            file.create_dataset('question_length', (0, ), maxshape=(None,), dtype='i')
            file.create_dataset('answer_length', (0, ), maxshape=(None,), dtype='i')

        # merge datapoints in data dictionary
        flat_data = defaultdict(list)
        for id in data.keys():
            for key, datapoint in data[id].items():
                flat_data[key].append(datapoint)

        # write out to file
        with h5py.File(self.preprocessed_file_path, 'a') as file:

            for key, value in flat_data.items():

                # get dataset
                h5py_dataset = file[key]

                # expand
                current_num_entries = h5py_dataset.shape[0]
                new_num_entries = len(value)
                if key in ['question', 'answer_input', 'answer_target']:
                    h5py_dataset.resize((current_num_entries+new_num_entries, self.max_utterance_length))
                elif key in ['question_length', 'answer_length']:
                    h5py_dataset.resize((current_num_entries+new_num_entries, ))

                # update
                h5py_dataset[current_num_entries:] = np.asarray(value)
