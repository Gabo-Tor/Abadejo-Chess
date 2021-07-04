import chess
import chess.svg
import numpy as np
import time
from valuators import *
from flask import Flask, Response, render_template, jsonify

board =  chess.Board()

def main():

  print(board)

  while not board.is_game_over():

    # human move
    print(board.legal_moves)
    nextMove = input()
    board.push(board.parse_san(nextMove))
    
    print("----------\nply: %d \n" %(board.ply()))
    startT = time.time()
    print(board)
    
    # Computer move
    nMove = np.argmin(list(moveValue(board, move) for move in board.legal_moves)) # this is probably not efficient
    nextMove = list(board.legal_moves)[nMove] 

    print("----------\nply: %d value: %f move: %s time: %f\n" %(board.ply()+1, moveValue(board, nextMove), nextMove, time.time()- startT))    
    board.push(nextMove)


    print(board)
    

app = Flask(__name__)

@app.route("/")
def init():
  return render_template('index.html')

@app.route("/board.svg")
def draw_board():
  if board.ply() >0:
    return Response(chess.svg.board(board, size= 600,lastmove= board.peek()), mimetype='image/svg+xml')
  else:
    return Response(chess.svg.board(board, size= 600), mimetype='image/svg+xml')

@app.route("/move")
def move():
  
  nMove = np.argmin(list(moveValue(board, move) for move in board.legal_moves)) # this is probably not efficient
  nextMove = list(board.legal_moves)[nMove] 
  board.push(nextMove)
  print("was here")

  return jsonify("location.reload();")

if __name__ == "__main__":
  app.run(debug= True)
  main()
