# Abadejo-Chess
Python Chess AI

## Use

runs in flask on localhost, go to http://127.0.0.1:5000/ for the move interface

## Dependancies

**Python 3** of course

* chess
* numpy
* flask
* time
* traceback
optinonal:
* pytorch


## Todo:

- [x] Piece value heuristic
- [x] MiniMax
- [x] Board rendering
- [x] Movility to the heuristic
- [x] MiniMax + Alpha Beta pruning
- [x] Piece square tables
- [ ] Move ordering (moves that capture pieces may be examined before moves that do not, and moves that have scored highly in earlier passes through the game-tree analysis may be evaluated before others)
- [x] Quiescence search
- [x] User input via submit field
- [ ] Info in the web and automatic enemy move
- [ ] User input via clicking
- [ ] Zobrist Hashing
- [ ] Tranposition tables
- [ ] Neural network evaluation
