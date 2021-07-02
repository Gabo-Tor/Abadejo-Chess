from math import inf
import re
import chess
import random as rn
import numpy as np

def heuristicValue(board):
  ## hand coded evaluation using features and domain knowledge

  # Using modern valuations for pieces
  materialValues = {chess.PAWN:   1,
                    chess.BISHOP: 3.33,
                    chess.KNIGHT: 3.05,
                    chess.ROOK:   5.63,
                    chess.QUEEN:  9.5,
                    chess.KING: 999}

  Bvalue, Wvalue, materialValue = 0,0,0

  for piece in materialValues.keys():
    Bvalue += board.pieces(piece, chess.BLACK).tolist().count(True)
    Wvalue += board.pieces(piece, chess.WHITE).tolist().count(True)

  # TODO: compute values for past, conected, isoleted pawns

  # we compute the real advantage, pieces are more valuble when there is less pieces
  materialValue = (Wvalue-Bvalue) / max(Wvalue, Bvalue)

  return materialValue

def moveValue(board, move):
  ## gives the value for a move in the given board using some evaluation
  board.push(move)
  value = minimax(board)
  board.pop()
  return value

def minimax(board, depth = 2):
  ## evaluates position to a depth using minimax

  if depth == 0:
    return heuristicValue(board)

  if board.turn == chess.WHITE:
    value = -np.inf
    for move in board.legal_moves:

      board.push(move)
      value = max(value, minimax(board, depth-1))
      board.pop()
    return value

  else:
    value = np.inf
    for move in board.legal_moves:
      board.push(move)
      value = min(value, minimax(board, depth-1))
      board.pop()
    return value

def main():
  board =  chess.Board()

  print(board)

  while not board.is_game_over():

    # human move
    print(board.legal_moves)
    nextMove = input()
    board.push(board.parse_san(nextMove))
    
    print("----------\nply: %d \n" %(board.ply()))
    print(board)

    # Computer move
    nMove = np.argmin(list(moveValue(board, move) for move in board.legal_moves)) # this is probably not efficient
    nextMove = list(board.legal_moves)[nMove] 

    print(nextMove)
    board.push(nextMove)

    print("----------\nply: %d \n" %(board.ply()))
    print(board)


if __name__ == "__main__":
  main()