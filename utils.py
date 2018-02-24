import io
import os
import json
import torch
import logging
import argparse
from torch.autograd import Variable
from collections import Counter, OrderedDict, defaultdict
class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2word(idx, i2w):

    words = str()

    for sent in idx:

        for id in sent:

            words += i2w[str(id)] + " "

        words += "\n"

    return words

def save_dial_to_json(prompts, replies, comment, root):

    dialogues = defaultdict(dict)

    prompts = prompts.split("\n")
    replies = replies.split("\n")

    for prompt, reply in zip(prompts, replies):
        dialogues[len(dialogues)]['prompt'] = prompt
        dialogues[len(dialogues)-1]['reply'] = reply

    if not os.path.exists(root):
        os.mkdir(root)

    file_path = os.path.join(root, "dial_"+comment+".json")
    with io.open(file_path, 'wb') as out:
        data = json.dumps(dialogues, ensure_ascii=False)
        out.write(data.encode('utf8', 'replace'))
