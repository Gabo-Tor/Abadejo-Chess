import chess
import chess.svg
import time
import traceback
from valuators import countMaterial, makeMove  # noqa
from flask import Flask, render_template, request, redirect, url_for  # noqa
from flask.wrappers import Response

# board =  chess.Board("r1b1k1nr/ppp2ppp/1bn1pq2/3p4/3P4/P3BN1P/1PP1PPPR/RNQ1KB2 b Qkq - 4 7")
# board =  chess.Board("5rk1/Ppppqp1p/6p1/4n3/4Q3/2P5/PP2NPPP/3RKB1R w Kq - 0 1")
# board =  chess.Board("r1b2rk1/ppppqp1p/6p1/4n3/4Q3/2P5/PP2NPPP/3RKB1R b Kq - 0 1")
useNeuralValuator = False
board = chess.Board()
moveList = []
moveTime, value, materialCount = 0, 0, 0

# board.push_san("e4")

app = Flask(__name__)


@app.route("/")
def init():
    if board.is_game_over():
        return render_template(
            "index.html",
            fen=board.fen().split(" ", 1)[0],
            moves=moveList,
            eval=value,
            moveTime="Game Over!!!",
            pCount=materialCount,
        )
    else:
        return render_template(
            "index.html",
            fen=board.fen().split(" ", 1)[0],
            moves=moveList,
            eval=value,
            moveTime=moveTime,
            pCount=materialCount,
        )


@app.route("/board.svg")
def draw_board():
    if board.ply() > 1:
        return Response(
            chess.svg.board(
                board,
                size=550,
                colors={
                    "square light": "#f0f0f0",
                    "square dark": "#8b9bad",
                    "margin": "#000000",
                    "coord": "#ffffff",
                },
                lastmove=board.peek(),
            ),
            mimetype="image/svg+xml",
        )
    else:
        return Response(
            chess.svg.board(
                board,
                size=550,
                colors={
                    "square light": "#f0f0f0",
                    "square dark": "#8b9bad",
                    "margin": "#000000",
                    "coord": "#ffffff",
                },
            ),
            mimetype="image/svg+xml",
        )


@app.route("/move", methods=["POST"])
def move():
    global value, moveTime, materialCount
    startT = time.time()
    nextMove, value = makeMove(board)
    value = round(value, 2)
    moveTime = round(time.time() - startT, 2)
    print(
        "----------\nply: %d value: %f move: %s time: %f\n"
        % (board.ply() + 1, value, nextMove, moveTime)
    )
    moveList.append(board.san(nextMove))
    board.push(nextMove)
    materialCount = countMaterial(board)
    return ""


@app.route("/human_move", methods=["POST"])
def human_move():
    startT = time.time()
    print("the recived txt is")
    print(request.form.get("hmove"))
    if "UCI" in str(request.form.get("hmove")):
        print(chess.Move.from_uci(str(request.form.get("hmove"))[-4:]))
        nextMove = chess.Move.from_uci(str(request.form.get("hmove"))[-4:])
    else:
        nextMove = board.parse_san(str(request.form.get("hmove")))
    # TODO: add feedback for ilegal moves
    if nextMove in board.legal_moves:
        try:
            print(
                "----------\nply: %d move: %s time: %f\n"
                % (board.ply() + 1, nextMove, time.time() - startT)
            )
            moveList.append(nextMove)
            board.push(nextMove)
            move()
        except AssertionError:
            traceback.print_exc()
    # make this more elegant, if a movement is a non explicit promotion, defaults to queen
    elif (
        chess.Move.from_uci(str(request.form.get("hmove"))[-4:] + "q")
        in board.legal_moves
    ):
        try:
            nextMove = chess.Move.from_uci(str(request.form.get("hmove"))[-4:] + "q")
            print(
                "----------\nply: %d move: %s time: %f\n"
                % (board.ply() + 1, nextMove, time.time() - startT)
            )
            moveList.append(nextMove)
            board.push(nextMove)
            move()
        except AssertionError:
            traceback.print_exc()

    return redirect("/")


if __name__ == "__main__":
    if useNeuralValuator:
        initNeuralValuator()
    app.run(debug=True)
