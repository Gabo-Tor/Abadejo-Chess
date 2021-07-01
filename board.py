import chess
import random as rn

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
    nextMove = rn.choice(list(board.legal_moves))
    board.push(nextMove)

    print("----------\nply: %d \n" %(board.ply()))
    print(board)



if __name__ == "__main__":
  main()