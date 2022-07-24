import chess
import time
import numpy as np
from chess.polyglot import zobrist_hash
from neural_valuator import NeuralValuator


materialValues = {
    chess.PAWN: 100,
    chess.BISHOP: 333,
    chess.KNIGHT: 305,
    chess.ROOK: 563,
    chess.QUEEN: 950,
    chess.KING: 9999,
}


# knigths on the rim are dim, this is 50% slower than using just the number
# fmt: off
pieceSquare = {
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
    chess.BISHOP: 64 * [3.33],
    chess.KNIGHT:
    [
        2.29,   2.44,   2.5925, 2.5925, 2.5925, 2.5925, 2.44,   2.29,
        2.44,   2.745,  3.05,   3.05,   3.05,   3.05,   2.745,  2.44,
        2.5925, 3.05,   3.2025, 3.355,  3.355,  3.2025, 3.05,   2.5925,
        2.5925, 3.2025, 3.355,  3.66,   3.66,   3.355,  3.2025, 2.5925,
        2.5925, 3.2025, 3.355,  3.66,   3.66,   3.355,  3.2025, 2.5925,
        2.5925, 3.05,   3.2025, 3.355,  3.355,  3.2025, 3.05,   2.5925,
        2.44,   2.745,  3.05,   3.2025, 3.2025, 3.05,   2.745,  2.44,
        2.29,   2.44,   2.5925, 2.5925, 2.5925, 2.5925, 2.44,   2.29
    ],

    chess.ROOK:  64 * [5.63],
    chess.QUEEN: 64 * [9.5],
    chess.KING: 
    [
        200, 201, 200, 200, 200, 200, 201, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 201, 200, 200, 200, 200, 201, 200
    ]}
# castling rigths usually are evaluated as one pawn in early game
# fmt: on


def countMaterial(board):

    # count material for each side using simple piece value

    simpleMaterialValues = {
        chess.PAWN: 1,
        chess.BISHOP: 3,
        chess.KNIGHT: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 200,
    }

    Bvalue, Wvalue = 0, 0
    for piece in simpleMaterialValues:
        for pos in board.pieces(piece, chess.BLACK):
            Bvalue += simpleMaterialValues[piece]
        for pos in board.pieces(piece, chess.WHITE):
            Wvalue += simpleMaterialValues[piece]

    return Wvalue - Bvalue


def heuristicValue(board):

    # hand coded evaluation using features and domain knowledge

    # Using modern valuations for pieces
    # TODO: piece square table reversible

    Bvalue, Wvalue, materialValue = 0, 0, 0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    else:

        for piece in materialValues.keys():

            # Bvalue += sum(map(lambda x,y:x*y, board.pieces(piece, chess.BLACK).tolist(), pieceSquare[piece] ))
            # Wvalue += sum(map(lambda x,y:x*y, board.pieces(piece, chess.WHITE).tolist(), pieceSquare[piece] ))
            # Bvalue += board.pieces(piece, chess.BLACK).tolist().count(True)* materialValues[piece]
            # Wvalue += board.pieces(piece, chess.WHITE).tolist().count(True)* materialValues[piece]
            # Bvalue += np.sum(np.multiply(board.pieces(piece, chess.BLACK).tolist(), pieceSquare[piece]))
            # Wvalue += np.sum(np.multiply(board.pieces(piece, chess.WHITE).tolist(), pieceSquare[piece]))

            # this is faster than combinations above
            for i, pos in enumerate(board.pieces(piece, chess.BLACK).tolist()):
                if pos:
                    Bvalue += pieceSquare[piece][i]

            for i, pos in enumerate(board.pieces(piece, chess.WHITE).tolist()):
                if pos:
                    Wvalue += pieceSquare[piece][i]

        # TODO: compute values for past, conected, isoleted pawns
        # https://es.wikipedia.org/wiki/Valor_relativo_de_las_piezas_de_ajedrez

        # we compute the real advantage, pieces are more valuble when there is less pieces
        materialValue = Wvalue - Bvalue

        # using pseudolegal instead of leag because is 30% faster and gives a ok result for this
        if board.turn == chess.WHITE:
            WMvalue = len(list(board.pseudo_legal_moves))
            board.turn = chess.BLACK
            BMvalue = len(list(board.pseudo_legal_moves))
            board.turn = chess.WHITE
        else:
            BMvalue = len(list(board.pseudo_legal_moves))
            board.turn = chess.WHITE
            WMvalue = len(list(board.pseudo_legal_moves))
            board.turn = chess.BLACK
        movilityValue = WMvalue - BMvalue
        # print(f"Material Val: {materialValue} Movility Val: {max(Wvalue, Bvalue) * movilityValue /20000}")
        return materialValue + max(Wvalue, Bvalue) * movilityValue / 20000


def simpleHeuristicValue(board):
    # very simple heuristic
    # Using modern valuations for pieces
    Bvalue, Wvalue = 0, 0

    for piece in materialValues:
        for pos in board.pieces(piece, chess.BLACK):
            Bvalue += materialValues[piece]
        for pos in board.pieces(piece, chess.WHITE):
            Wvalue += materialValues[piece]

    return Wvalue - Bvalue


def makeMove(board):
    global valueTable
    global posEvaluated
    global hashedPos
    maxDepth = 5
    sTime = time.perf_counter()
    posEvaluated, hashedPos = 0, 0
    moveValues = []
    valueTable = dict()

    for depth in range(maxDepth):
        print(f"\n\u001b[36m-Evaluating at depth: {depth}\u001b[0m ", end="")
        moveValues = list(
            moveValue(board, move, depth=depth, maxDepth=depth)
            for move in board.legal_moves
        )
        print(f"Time spent: {time.perf_counter()-sTime:.2f}s")
        print(f"Evaluated {len(valueTable)} pos | Hashed {hashedPos} pos")
        print(f"    ↖ Percentage: {100*hashedPos/len(valueTable):.2f}% ↗")

        if (time.perf_counter() - sTime) > 60:
            break

    if board.turn == chess.WHITE:
        idxMove = np.argmax(moveValues)
    else:
        idxMove = np.argmin(moveValues)
    nextMove = list(board.legal_moves)[idxMove]

    for val, move in zip(moveValues, board.legal_moves):
        print(move, ":", f"{val / 100:.2f}")

    return nextMove, moveValues[idxMove] / 100


def moveValue(board, move, depth=0, maxDepth=0):
    # gives the value for a move in the given board using some evaluation

    board.push(move)
    value = negamaxAB(board, depth=depth, maxDepth=maxDepth)
    board.pop()

    return value


def negamaxAB(board, depth=0, maxDepth=0, alpha=-np.inf, beta=np.inf, color=1):
    # TODO esto está mal implementado rehacer
    boardHash = zobrist_hash(board)
    global hashedPos

    if boardHash in valueTable:
        if valueTable[boardHash]["depth"] >= depth:
            hashedPos += 1
            return valueTable[boardHash]["value"]

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
    ):
        value = 0
        valueTable[boardHash] = {"depth": depth, "value": value}
        return value

    elif depth == 0:
        value = color * simpleHeuristicValue(board)
        valueTable[boardHash] = {"depth": depth, "value": value}
        return value

    value = -np.inf

    for move in board.legal_moves:
        board.push(move)
        value = max(
            value,
            -negamaxAB(
                board,
                depth - 1,
                maxDepth=maxDepth,
                alpha=-beta,
                beta=-alpha,
                color=-color,
            ),
        )
        board.pop()

        alpha = max(alpha, value)

        if alpha >= beta:
            break

    valueTable[boardHash] = {"depth": alpha, "value": value}
    return alpha


def negamaxHash(board, depth=0, maxDepth=0, color=1):
    # Negamax using zoobrist hash
    boardHash = zobrist_hash(board)
    global hashedPos

    if boardHash in valueTable:
        if valueTable[boardHash]["depth"] >= depth:
            hashedPos += 1
            return valueTable[boardHash]["value"]

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
    ):
        value = 0
        valueTable[boardHash] = {"depth": depth, "value": value}
        return value

    elif depth == 0:
        value = color * simpleHeuristicValue(board)
        valueTable[boardHash] = {"depth": depth, "value": value}
        return value

    value = -np.inf

    for move in board.legal_moves:
        board.push(move)
        value = max(
            value, -negamaxHash(board, depth - 1, maxDepth=maxDepth, color=-color)
        )
        board.pop()

    valueTable[boardHash] = {"depth": depth, "value": value}
    return value


def negamax(board, depth=0, maxDepth=0, color=1):
    # simple negamax zoobrist hash is not used, just computed so times are compararble
    boardHash = zobrist_hash(board)
    global hashedPos

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
    ):
        value = 0
        valueTable[boardHash] = {"depth": depth, "value": value}
        return value

    elif depth == 0:
        value = color * simpleHeuristicValue(board)
        valueTable[boardHash] = {"depth": depth, "value": value}
        return value

    value = -np.inf

    for move in board.legal_moves:
        board.push(move)
        value = max(value, -negamax(board, depth - 1, maxDepth=maxDepth, color=-color))
        board.pop()

    valueTable[boardHash] = {"depth": depth, "value": value}
    return value


def initNeuralValuator():
    # TODO This is an ugly way of doing this
    global neuralValuator
    neuralValuator = NeuralValuator()


if __name__ == "__main__":
    # this is a test
    board = chess.Board()
    neuralValuator = NeuralValuator()
    print(neuralValuator.NeuralValue(board))
