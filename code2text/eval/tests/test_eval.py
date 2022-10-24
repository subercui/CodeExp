import sys
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_folder))

from eval import kendalls_tau


def test_eval():
    assert True
