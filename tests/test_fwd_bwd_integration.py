import pytest

import setup_tests
import sys; print(sys.path)
from symbolicmathematics.integrate import Integrator

@pytest.fixture
def integrator():
    return Integrator(cpu=True, model_name='fwd_bwd')

def test_latex(integrator):
    assert integrator.integrate(r'\sin(x)') == r'1 - \cos{\left(x \right)}'

def test_pyexpr(integrator):
    assert integrator.integrate('sin(x)') == r'1 - \cos{\left(x \right)}'
