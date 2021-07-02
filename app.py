import chess
import chess.svg
import numpy as np
from valuators import *
from flask import Flask, Response

board =  chess.Board()

def main():

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
    

app = Flask(__name__)

@app.route("/")
def init():
  return '<html><body><img src="board.svg"></img></html></body>'

@app.route("/board.svg")
def draw_board():
  if board.ply() >0:
    return Response(chess.svg.board(board, size= 600,lastmove= board.peek()), mimetype='image/svg+xml')
  else:
    return Response(chess.svg.board(board, size= 600), mimetype='image/svg+xml')

@app.route("/move", methods=["GET"])
def move():
  nMove = np.argmin(list(moveValue(board, move) for move in board.legal_moves)) # this is probably not efficient
  nextMove = list(board.legal_moves)[nMove] 
  board.push(nextMove)
  return ""

if __name__ == "__main__":
  app.run()
  main()