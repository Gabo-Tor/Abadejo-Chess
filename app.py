import chess
import chess.svg
import numpy as np
import time
import traceback
from valuators import *
from flask import Flask, Response, render_template, request, redirect

# board =  chess.Board("r1b1k1nr/ppp2ppp/1bn1pq2/3p4/3P4/P3BN1P/1PP1PPPR/RNQ1KB2 b Qkq - 4 7")
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
  startT = time.time()

  moveValues = list(moveValue(board, move) for move in board.legal_moves)
  print(board.legal_moves)
  print(moveValues)

  if board.turn == chess.WHITE:
    nMove = np.argmax(moveValues) # this is probably not efficient
  else:
    nMove = np.argmin(moveValues) # this is probably not efficient
  nextMove = list(board.legal_moves)[nMove] 


  print("----------\nply: %d value: %f move: %s time: %f\n" %(board.ply()+1, moveValue(board, nextMove), nextMove, time.time()- startT))    
  board.push(nextMove)
  return ""

@app.route("/human_move", methods=['POST'])
def human_move():
  startT = time.time()
  print("the recived txt is")
  print(request.form.get('hmove'))
  nextMove = request.form.get('hmove')
  try:
    print("----------\nply: %d value: %f move: %s time: %f\n" %(board.ply()+1, moveValue(board, board.parse_san(nextMove)), board.parse_san(nextMove), time.time()- startT))    
    board.push(board.parse_san(nextMove))
    move()
  except:
    traceback.print_exc()

  return redirect('/')

if __name__ == "__main__":
  initNeuralValuator()
  app.run(debug= True)
  # main()
