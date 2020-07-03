import os
import json
import pathlib
from typing import Optional

import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import torch

from src.utils import AttrDict
from src.envs import build_env
from src.model import build_modules
from src.utils import to_cuda
from src.envs.sympy_utils import simplify

project_root = pathlib.Path(__file__).parent

def load_settings():
    local_settings = json.load((project_root / 'local_settings.json').open())
    default_settings = json.load((project_root / 'default_settings.json').open())

    params = default_settings
    params.update(local_settings)
    params['reload_model'] = str(project_root / 'dumped' / 'fwd_bwd_ibp.pth')
    return AttrDict(params)

params = load_settings()

env = build_env(params)

x = env.local_dict['x']

modules = build_modules(env, params)
encoder = modules['encoder']
decoder = modules['decoder']

def to_tensor(expr):
    prefix = env.sympy_to_prefix(expr)
    cleaned = env.clean_prefix(['sub', 'derivative', 'f', 'x', 'x'] + prefix)
    tensor = torch.LongTensor(
        [env.eos_index] + 
        [env.word2id[w] for w in cleaned] + 
        [env.eos_index]
    ).view(-1, 1)
    length = torch.LongTensor([len(tensor)])
    
    return to_cuda([tensor, length], cpu=params.cpu)

def generate_hypotheses(tensor, length):
    with torch.no_grad():
        encoded = encoder('fwd', x=tensor, lengths=length, causal=False).transpose(0, 1)
        _, _, beam = decoder.generate_beam(encoded, length, beam_size=10, 
                                           length_penalty=1.0, early_stopping=1, max_len=200)
    return beam[0].hyp

def choose_hypothesis(expr, hypotheses):
    def first(x): return x[0]
    for score, sent in sorted(hypotheses, key=first, reverse=True):
        ids = sent[1:].tolist()
        tokens = [env.id2word[wid] for wid in ids]
        
        hypothesis = env.infix_to_sympy(env.prefix_to_infix(tokens)) 
        
        if simplify(hypothesis.diff(x) - expr, seconds=1) == 0:
            return hypothesis
        
        return None

def integrate(f):
    expr = parse_latex(f)
    tensor, length = to_tensor(expr)
    hypothesis = generate_hypotheses(tensor, length)
    return choose_hypothesis(expr, hypothesis)