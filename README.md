# Abadejo-Chess
Python Chess AI

## Use

runs in flask on localhost, for now go to http://127.0.0.1:5000/move for a move, then refresh http://127.0.0.1:5000/ i know, super intuitive

## Dependancies

Python 3 of course

* chess
* numpy
* flask
* time

## Todo:

* add movility to the heuristic
* add move ordering (moves that capture pieces may be examined before moves that do not, and moves that have scored highly in earlier passes through the game-tree analysis may be evaluated before others)
* add quiescence search
* add tranposition table with Zobrist Hashing
