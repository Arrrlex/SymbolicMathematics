from typing import Optional

from sympy.parsing.latex import parse_latex
import sympy as sp
import torch
from func_timeout import FunctionTimedOut

from src.utils import to_cuda, load_settings
from src.envs import build_env
from src.model import build_modules
from src.envs.sympy_utils import simplify
from src.envs.char_sp import InvalidPrefixExpression

def preprocess(expr):
    e = sp.S('e')
    exponential_subexprs = [subexpr 
            for subexpr in sp.preorder_traversal(expr)
            if subexpr.func == sp.Pow and subexpr.base == e]
    fixed_subexprs = [sp.exp(subexpr.exp) for subexpr in exponential_subexprs]

    fixed_expr = expr
    for (old, new) in zip(exponential_subexprs, fixed_subexprs):
        fixed_expr = fixed_expr.subs(old, new)

    return fixed_expr

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
            expr = parse_latex(f)
            expr_fixed = preprocess(expr)
            antiderivative = self.integrate_sympy(expr_fixed)
        except Exception as e:
            print("Error thrown assuming latex:", e)
            try:
                expr = sp.sympify(f, locals=self.env.local_dict)
                print(sp.srepr(expr))
                antiderivative = self.integrate_sympy(expr_fixed)
            except Exception as e:
                print("Error thrown assuming python:", e)
                return None
        return sp.latex(antiderivative)

    def integrate_pyexpr(self, f):
        expr = sp.sympify(f, locals=self.env.local_dict)
        return self.integrate_sympy(expr)

    def integrate_latex(self, f):
        expr = parse_latex(f)
        expr_fixed = preprocess(expr)
        return self.integrate_sympy(expr_fixed)
    
    def integrate_sympy(self, expr):
        tensor, length = self.to_tensor(expr)
        hypotheses = self.generate_hypotheses(tensor, length)
        return self.choose_hypothesis(expr, hypotheses)
