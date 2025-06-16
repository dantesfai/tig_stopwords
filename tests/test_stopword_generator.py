# tests/test_stopword_generator.py
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stopword_generator import StopwordGenerator

def test_generate_stopwords():
    stopword_gen = StopwordGenerator(corpus_files=["examples/example_input.txt"])
    stopwords = stopword_gen.generate_stop_words(threshold=2.5)
    assert len(stopwords) > 0
    assert isinstance(stopwords[0], tuple)
