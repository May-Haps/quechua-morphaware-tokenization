"""
Usage:
    python scripts/visualize_fst.py <word> [word2 ...]
    python scripts/visualize_fst.py  # reads words from stdin, one per line
"""
import sys
import os

# allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.process_word_windows import run_fst


def show(word: str) -> None:
    morphemes = run_fst(word)
    if morphemes is None:
        print(f'{word!r:30s}  -> <no FST match>')
    else:
        print(f'{word!r:30s}  -> {" + ".join(morphemes)}')


if __name__ == '__main__':
    words = sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()]
    if not words:
        print('provide words as arguments or via stdin')
        sys.exit(1)
    for w in words:
        show(w)
