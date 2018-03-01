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

def idx2word(idx, i2w, pad_idx):
    """Maps given indicies `idx` to words.

    Parameters
    ----------
    idx : torch.LongTensor (or any other two iterable wrapping an iterable)
        Word indicies.
    i2w : dict
        Dictionary mapping indicies to words.
    pad_idx : int
        Index of padding token.

    Returns
    -------
    string
        A single string containing the words encoded by the given indicies. The individual
        sequences are seperated by `\n`.

    """

    words = str()

    for sent in idx:
        for id in sent:
            if id == pad_idx:
                break
            words += i2w[str(id)] + " "

        words += "\n"

    return words

def save_dial_to_json(prompts, replies, comment, root):
    """Saves the prompts and replies string to a json file.

    Parameters
    ----------
    prompts : string
        String of prompts, seperated by `\n`.
    replies : list(string)
        String of replies, seperated by `\n`.
    comment : string
        This string will be prepended to the json.
    root : string
        Directory of output file. If it does not exist, it will be created.

    """

    dialogues = defaultdict(OrderedDict)

    prompts = prompts.split("\n")
    for pi, prompt in enumerate(prompts):
        id = len(dialogues)
        dialogues[id]['prompt'] = prompt
        for ri, reply in enumerate(replies):
            reply = reply.split("\n")[pi]
            dialogues[id]['reply'+str(ri)] = reply

    if not os.path.exists(root):
        os.mkdir(root)

    file_path = os.path.join(root, "dial_"+comment+".json")
    with io.open(file_path, 'wb') as out:
        data = json.dumps(dialogues, ensure_ascii=False)
        out.write(data.encode('utf8', 'replace'))

def experiment_name(args):

    exp_name = str()

    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR=%f_"%args.learning_rate
    exp_name += "KLA=%i_"%(1 if args.kl_anneal else 0)
    if args.kl_anneal:
        exp_name += "KLk=%f_"%args.kla_k
        exp_name += "KLx0=%f_"%args.kla_x0
    exp_name += "bi=%i_"%(1 if args.bidirectional_encoder else 0)
    exp_name += "EMB=%i_"%args.embedding_size
    exp_name += "HID=%i_"%args.hidden_size
    exp_name += "Z=%i_"%args.latent_size
    exp_name += "WD=%f"%args.word_dropout

    return exp_name
