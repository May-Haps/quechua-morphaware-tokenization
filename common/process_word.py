"""
Input: quechua word (string)
Output: quechua word split by morphemes, may have unneccessary additional text to parse through
TODO: Verify parsing returns one parse, and remove "=", split on "+"
"""

import pynini

def run_fst(word):
    try:
        # 2. Convert the word into an acceptor (a simple FST)
        # 3. Compose (@) the word with the FST
        # 4. Find the shortest path and convert back to a string
        fst = pynini.Fst.read("analyzeCuzco.fst")
        lattice = word @ fst
        return pynini.shortestpath(lattice).string()
    except pynini.FstOpError:
        return "<No Match Found>"

