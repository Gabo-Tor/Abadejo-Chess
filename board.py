import chess
import random as rn

def main():
  board =  chess.Board()


  print(board)

  print(board.legal_moves)

  while not board.is_game_over():

    nextMove = rn.choice(list(board.legal_moves))
    board.push(nextMove)
    print("----------\nply: %d \n" %(board.ply()))
    print(board)


if __name__ == "__main__":
  main()