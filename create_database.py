from typing import Type
import chess
import chess.pgn
import numpy as np

# we usea a PGN database from: https://database.lichess.org/
DATABASE = r"C:\lichess_db_standard_rated_2015-05.pgn"

def read():
  # Reads a lichess PGN dump and saves all anotated positions to a database file 
  positions, games, exponent = 0, 0, 1
  dbX = np.array(parse_board(chess.Board())) # Data
  dbY = np.array([520]) # Target
  pgn = open(DATABASE)

  for games in range(2137556): # Don't really now how to read the length of the DATABASE
    game = chess.pgn.read_game(pgn)

    for move in game.mainline():
      povScore = move.eval()
      if povScore == None: # We only care about anotated games
        continue
      board= move.board()
      dbX = np.append(dbX, parse_board(board), axis=0)
      # we should be encoding turn info or evaluating from the corresponig sire it tis wrong doing it like this
      dbY = np.append(dbY, [povScore.pov(chess.WHITE).wdl(model="lichess").wins], axis=0)
      positions += 1
    games += 1

    if not games % 100:
      print(f"positions evaluated: {positions} in {games} games")

    if not games % (10**exponent):
      np.save(f"{positions}_positions_data",dbX)
      np.save(f"{positions}_positions_targets",dbY)
      dbX = np.array(parse_board(chess.Board())) # Reset data array
      dbY = np.array([520]) # Reset target array
      exponent += 1
  pgn.close()


def parse_board(board):
  # Returns a 7x8x8 one hot encoded piece location tensor, with 1 and -1 
  # corresponing to black and white and an extra map set to 1
  pieceTensor = {
  chess.PAWN:   np.zeros((1,1,8,8), dtype= 'int8'),
  chess.BISHOP: np.zeros((1,1,8,8), dtype= 'int8'),
  chess.KNIGHT: np.zeros((1,1,8,8), dtype= 'int8'),
  chess.ROOK:   np.zeros((1,1,8,8), dtype= 'int8'),
  chess.QUEEN:  np.zeros((1,1,8,8), dtype= 'int8'),
  chess.KING:   np.zeros((1,1,8,8), dtype= 'int8')}

  for square in range(64):
    piece = board.piece_map().get(square)
    if not piece == None:
  # Our data is really sparse, so we encode both colors in the same map
      if piece.color == chess.WHITE:
        pieceTensor.get(piece.piece_type)[0, 0, square//8, square%8] = 1
      else:
        pieceTensor.get(piece.piece_type)[0, 0, square//8, square%8] = -1
  # We add a board full of ones so we don't loose track of the borders of the board when using padding
  outTensor = np.ones((1,1,8,8), dtype= 'int8') 
  for piece in pieceTensor.keys():
    outTensor = np.append(outTensor, pieceTensor.get(piece),axis = 1)
  return outTensor


if __name__ == "__main__":
  read()
