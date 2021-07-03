import chess
import numpy as np

materialValues = {chess.PAWN:   100,
                  chess.BISHOP: 333,
                  chess.KNIGHT: 305,
                  chess.ROOK:   563,
                  chess.QUEEN:  950,
                  chess.KING: 9999}

        

# knigths on the rim are dim
pieceSquare =  {

chess.PAWN: 64* [1.0],
chess.BISHOP: 64* [3.33],
chess.KNIGHT: 
[
    2.29,  2.44, 2.5925, 2.5925, 2.5925, 2.5925, 2.44, 2.29,
    2.44,  2.745, 3.05,  3.05,   3.05,   3.05,   2.745, 2.44,
    2.5925, 3.05, 3.2025, 3.355, 3.355, 3.2025, 3.05, 2.5925,
    2.5925, 3.2025, 3.355, 3.66, 3.66, 3.355, 3.2025, 2.5925,
    2.5925, 3.2025, 3.355, 3.66, 3.66, 3.355, 3.2025, 2.5925,
    2.5925, 3.05, 3.2025, 3.355, 3.355, 3.2025, 3.05, 2.5925,
    2.44,  2.745, 3.05,   3.2025, 3.2025, 3.05, 2.745, 2.44,
    2.29,  2.44, 2.5925,  2.5925, 2.5925, 2.5925, 2.44, 2.29
],

chess.ROOK:  64* [5.63],
chess.QUEEN: 64* [9.5],
chess.KING: 64 * [9999]
 }

def heuristicValue(board):
  ## hand coded evaluation using features and domain knowledge

  # Using modern valuations for pieces
  # TODO: piece square table reversible

  Bvalue, Wvalue, materialValue = 0,0,0

  for piece in materialValues.keys():

    Bvalue += sum(map(lambda x,y:x*y, board.pieces(piece, chess.BLACK).tolist(), pieceSquare[piece] ))
    Wvalue += sum(map(lambda x,y:x*y, board.pieces(piece, chess.WHITE).tolist(), pieceSquare[piece] ))

  # TODO: compute values for past, conected, isoleted pawns 
  # https://es.wikipedia.org/wiki/Valor_relativo_de_las_piezas_de_ajedrez

  # we compute the real advantage, pieces are more valuble when there is less pieces
  materialValue = (Wvalue-Bvalue) / max(Wvalue, Bvalue)


  if board.is_checkmate():
      if board.turn:
          return -99999
      else:
          return 99999
  if board.is_stalemate():
      return 0
  if board.is_insufficient_material():
      return 0
  else:
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
