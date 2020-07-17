
import traceback

from sympy.parsing.latex import parse_latex
import sympy as sp
import torch
from func_timeout import FunctionTimedOut

from .utils import to_cuda, load_settings
from .envs import build_env
from .model import build_modules
from .envs.sympy_utils import simplify
from .envs.char_sp import InvalidPrefixExpression

class Integrator(object):
    def __init__(self, cpu, model_name=None, model_path=None, **kwargs):
        self.params = load_settings(cpu=cpu, model_name=model_name, model_path=model_path, **kwargs)
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
        beam_size = 10

        with torch.no_grad():
            encoded = self.encoder('fwd', x=tensor, lengths=length, causal=False).transpose(0, 1)
            _, _, beam = self.decoder.generate_beam(encoded, length, beam_size=beam_size, 
                                            length_penalty=1.0, early_stopping=1, max_len=200)
        hypotheses = beam[0].hyp
        assert len(hypotheses) == beam_size
        return hypotheses

    def choose_hypothesis(self, expr, hypotheses):
        def first(x): return x[0]
        for score, sent in sorted(hypotheses, key=first, reverse=True):
            ids = sent[1:].tolist()
            tokens = [self.env.id2word[wid] for wid in ids]
            
            try:
                infix = self.env.prefix_to_infix(tokens)
            except InvalidPrefixExpression:
                continue

            try:
                hypothesis = self.env.infix_to_sympy(infix)
            except ValueErrorExpression:
                continue

            try:
                if simplify(hypothesis.diff(self.env.local_dict['x']) - expr, seconds=1) == 0:
                    return simplify(hypothesis, seconds=1)
            except FunctionTimedOut:
                continue
            
        print("No hypothesis was correct.")
        return None

    def integrate(self, f):
        try:
            antiderivative = self.integrate_latex(f)
        except:
            print("Error when interpreting as latex")
            traceback.print_exc()

            try:
                antiderivative = self.integrate_pyexpr(f)
            except:
                print("Error when interpreting as pyexpr")
                traceback.print_exc()
                return None

        try:
            return sp.latex(antiderivative)
        except:
            return None

    def integrate_pyexpr(self, f):
        expr = sp.sympify(f, locals=self.env.local_dict)
        return self.integrate_sympy(expr)

    def integrate_latex(self, f):
        expr = parse_latex(f)
        
        for var_name, variable in self.env.variables.items():
            expr = expr.subs(sp.Symbol(var_name), variable)

        for func_name, function in self.env.functions.items():
            expr = expr.subs(sp.Function(func_name), function)
        
        for coeff_name, coefficient in self.env.coefficients.items():
            expr = expr.subs(sp.Symbol(coeff_name), coefficient)

        return self.integrate_sympy(expr)
    
    def integrate_sympy(self, expr):
        tensor, length = self.to_tensor(expr)
        hypotheses = self.generate_hypotheses(tensor, length)
        return self.choose_hypothesis(expr, hypotheses)
