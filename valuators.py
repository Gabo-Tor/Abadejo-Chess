import chess
import numpy as np
from neural_valuator import *
from chess.polyglot import zobrist_hash

materialValues = {chess.PAWN:   100,
                  chess.BISHOP: 333,
                  chess.KNIGHT: 305,
                  chess.ROOK:   563,
                  chess.QUEEN:  950,
                  chess.KING: 9999}

    
# knigths on the rim are dim, this is 50% slower than using just the number
pieceSquare =  {

chess.PAWN: 
[
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.05, 1.10, 1.15, 1.15, 0.90, 1.00, 1.00,
    1.00, 1.00, 1.20, 1.40, 1.40, 0.90, 1.00, 1.00,
    1.00, 1.00, 1.20, 1.40, 1.40, 0.90, 1.00, 1.00,
    1.00, 1.05, 1.10, 1.15, 1.15, 0.90, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
],
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
chess.KING:  
[
    200,  201, 200, 200, 200, 200, 201, 200,
    200,  200, 200, 200, 200, 200, 200, 200,
    200,  200, 200, 200, 200, 200, 200, 200,
    200,  200, 200, 200, 200, 200, 200, 200,
    200,  200, 200, 200, 200, 200, 200, 200,
    200,  200, 200, 200, 200, 200, 200, 200,
    200,  200, 200, 200, 200, 200, 200, 200,
    200,  201, 200, 200, 200, 200, 201, 200
] }
# castling rigths usually are evaluated as one pawn in early game

def heuristicValue(board):

  ## hand coded evaluation using features and domain knowledge

  # Using modern valuations for pieces
  # TODO: piece square table reversible

  Bvalue, Wvalue, materialValue = 0,0,0

  for piece in materialValues.keys():

    Bvalue += sum(map(lambda x,y:x*y, board.pieces(piece, chess.BLACK).tolist(), pieceSquare[piece] ))
    Wvalue += sum(map(lambda x,y:x*y, board.pieces(piece, chess.WHITE).tolist(), pieceSquare[piece] ))
    # Bvalue += board.pieces(piece, chess.BLACK).tolist().count(True)* materialValues[piece]
    # Wvalue += board.pieces(piece, chess.WHITE).tolist().count(True)* materialValues[piece]

  # TODO: compute values for past, conected, isoleted pawns 
  # https://es.wikipedia.org/wiki/Valor_relativo_de_las_piezas_de_ajedrez

  # we compute the real advantage, pieces are more valuble when there is less pieces
  materialValue = (Wvalue-Bvalue) 

  if board.is_stalemate() or board.is_insufficient_material():
    return 0
  else:

    if board.turn == chess.WHITE:
      WMvalue = len(list(board.legal_moves))
      board.turn = chess.BLACK
      BMvalue = len(list(board.legal_moves))
      board.turn = chess.WHITE
    else:
      BMvalue = len(list(board.legal_moves))
      board.turn = chess.WHITE
      WMvalue = len(list(board.legal_moves))
      board.turn = chess.BLACK
    movilityValue = WMvalue - BMvalue
    # print(f"Material Val: {materialValue} Movility Val: {max(Wvalue, Bvalue) * movilityValue /20000}")
    return materialValue + max(Wvalue, Bvalue) * movilityValue /20000

def makeMove(board):
  global valueTable
  global posEvaluated
  posEvaluated = 0
  valueTable = [dict() for x in range(8)]
  moveValues = list(moveValue(board, move) for move in board.legal_moves)


  if board.turn == chess.WHITE:
    nMove = np.argmax(moveValues) # this is probably not efficient
  else:
    nMove = np.argmin(moveValues) # this is probably not efficient
  nextMove = list(board.legal_moves)[nMove] 

  print(f"evaluated {posEvaluated} positions, hashed {len(valueTable[0])} positions")

  return nextMove, moveValue(board, nextMove)


def moveValue(board, move):
  global valueTable
  ## gives the value for a move in the given board using some evaluation

  board.push(move)
  value = minimaxAB(board)
  board.pop()

  return value

def minimax(board, depth = 2):
  ## evaluates position to a depth using minimax

  if depth == 0:
    return neuralValuator.NeuralValue(board)
    # return heuristicValue(board)

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



def minimaxAB(board, depth = 2, alpha = -np.inf, beta = np.inf):
  global valueTable
  boardHash = zobrist_hash(board)
  ## evaluates position to a depth using minimax with alpha beta pruning
  global posEvaluated
  posEvaluated += 1
  if boardHash in valueTable[depth]:
    
    return valueTable[depth][boardHash]

  else:
    if depth == 0 or board.is_stalemate() or board.is_insufficient_material():
      if len(list(move for move in board.legal_moves if board.is_capture(move)))>0: #  node is not quiet
        
        value = quiescence_search(board, depth = 0, alpha = alpha, beta =  beta)
        valueTable[depth][boardHash] = value
        return value
      else:
        # value = neuralValuator.NeuralValue(board)
        value = heuristicValue(board)
        valueTable[depth][boardHash] = value
        return value
        

    if board.turn == chess.WHITE:
      value = -np.inf
      for move in board.legal_moves:
        if board.is_capture(move):
          board.push(move)
          value = max(value, minimaxAB(board, depth-1, alpha = alpha, beta =  beta))
          board.pop()

          if value>= beta:
            break
          alpha = max(alpha, value)

      for move in board.legal_moves:
        if not board.is_capture(move):
          board.push(move)
          value = max(value, minimaxAB(board, depth-1, alpha = alpha, beta =  beta))
          board.pop()

          if value>= beta:
            break
          alpha = max(alpha, value)

    else:
      value = np.inf
      for move in board.legal_moves:
        if board.is_capture(move):
          board.push(move)
          value = min(value, minimaxAB(board, depth-1, alpha = alpha, beta =  beta))
          board.pop()

          if value<= alpha:
            break
          beta = min(beta, value)

      for move in board.legal_moves:
        if not board.is_capture(move):
          board.push(move)
          value = min(value, minimaxAB(board, depth-1, alpha = alpha, beta =  beta))
          board.pop()

          if value<= alpha:
            break
          beta = min(beta, value)

    valueTable[depth][boardHash] = value
    return value


def quiescence_search(board, depth = 99, alpha = -np.inf, beta = np.inf):
  ## evaluates position to a depth using quiescence search

  if len(list(move for move in board.legal_moves if board.is_capture(move)))== 0\
     or board.is_stalemate() or board.is_insufficient_material() or depth == 0: #  node is quiet
    # return neuralValuator.NeuralValue(board)
    return heuristicValue(board)

  else:      

    if board.turn == chess.WHITE:
      value = -np.inf
      for move in board.legal_moves:
        if board.is_capture(move):
          board.push(move)
          value = max(value, quiescence_search(board, depth-1, alpha = alpha, beta =  beta))
          board.pop()

          if value>= beta:
            break
          alpha = max(alpha, value)

      return value

    else:
      value = np.inf
      for move in board.legal_moves:
        if board.is_capture(move):
          board.push(move)
          value = min(value, quiescence_search(board, depth-1, alpha = alpha, beta =  beta))
          board.pop()

          if value<= alpha:
            break
          beta = min(beta, value)

      return value

def initNeuralValuator():
  # This is an ugly way of doing this
    global neuralValuator
    neuralValuator = NeuralValuator()


if __name__ == "__main__":
  # this is a test
  board = chess.Board()
  neuralValuator = NeuralValuator()
  print(neuralValuator.NeuralValue(board))