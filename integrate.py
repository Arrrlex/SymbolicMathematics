import json
import pathlib
from typing import Optional

from sympy.parsing.latex import parse_latex
import torch

from src.utils import AttrDict, to_cuda
from src.envs import build_env
from src.model import build_modules
from src.envs.sympy_utils import simplify

project_root = pathlib.Path(__file__).parent

def load_settings(**kwargs):
    default_settings = json.load((project_root / 'default_settings.json').open())
    params = default_settings
    params.update(kwargs)
    return AttrDict(params)

class Integrator(object):
    def __init__(self, cpu, model_path):
        self.params = load_settings(cpu=cpu, reload_model=model_path)
        self.env = build_env(self.params)
        
        self.modules = build_modules(self.env, self.params)
        self.encoder = self.modules['encoder']
        self.decoder = self.modules['decoder']


    def to_tensor(self, expr):
        prefix = self.env.sympy_to_prefix(expr)
        cleaned = self.env.clean_prefix(['sub', 'derivative', 'f', 'x', 'x'] + prefix)
        tensor = torch.LongTensor(
            [self.env.eos_index] + 
            [self.env.word2id[w] for w in cleaned] + 
            [self.env.eos_index]
        ).view(-1, 1)
        length = torch.LongTensor([len(tensor)])
        
        return to_cuda([tensor, length], cpu=self.params.cpu)

    def generate_hypotheses(self, tensor, length):
        with torch.no_grad():
            encoded = self.encoder('fwd', x=tensor, lengths=length, causal=False).transpose(0, 1)
            _, _, beam = self.decoder.generate_beam(encoded, length, beam_size=10, 
                                            length_penalty=1.0, early_stopping=1, max_len=200)
        return beam[0].hyp

    def choose_hypothesis(self, expr, hypotheses):
        def first(x): return x[0]
        for score, sent in sorted(hypotheses, key=first, reverse=True):
            ids = sent[1:].tolist()
            tokens = [self.env.id2word[wid] for wid in ids]
            
            infix = self.env.prefix_to_infix(tokens)
            hypothesis = self.env.infix_to_sympy(infix)
            
            if simplify(hypothesis.diff(self.env.local_dict['x']) - expr, seconds=1) == 0:
                return hypothesis
            
            return None

    def integrate(self, f):
        expr = parse_latex(f)
        tensor, length = self.to_tensor(expr)
        hypothesis = self.generate_hypotheses(tensor, length)
        return self.choose_hypothesis(expr, hypothesis)