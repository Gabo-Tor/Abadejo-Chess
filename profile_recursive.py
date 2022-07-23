# script intended for profiling recursive algorithms
from valuators import makeMove, chess

board = chess.Board("r1b2rk1/ppppqp1p/6p1/4n3/4Q3/2P5/PP2NPPP/3RKB1R b Kq - 0 1")
print(makeMove(board))
# cProfile.run('makeMove(board)')

"""
to use:

python -m cProfile -o program.prof profile_recursive.py
snakeviz program.prof

"""
