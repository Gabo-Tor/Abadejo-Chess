import torch
from torch import nn
from create_database import *
import numpy as np
import torch.nn.functional as F
import chess

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.convA1 = nn.Conv2d(7, 32, 3, stride= 1, padding= 1)
        self.convA2 = nn.Conv2d(32, 32, 3, stride= 1, padding= 0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1152, 64)
        self.fc2 = nn.Linear(64, 1)
        

    def forward(self, x):
        x = self.convA1(x)
        x = self.convA2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output

class NeuralValuator():
  def __init__(self):
    # device = torch.device('cpu')
    self.device = "cpu"
    self.model = NeuralNetwork()
    self.model = torch.load('model.pt', map_location=self.device)

  def NeuralValue(self, board):



    if board.turn == chess.WHITE: # We always look the board from the perspective of the player about to move
      state = parse_board(board)
      self.model.eval()
      with torch.no_grad():
        pred = self.model(torch.from_numpy(np.divide(state,1000)).float())
      return pred


    else:
      state = parse_board(board.mirror())
      self.model.eval()
      with torch.no_grad():
        pred = self.model(torch.from_numpy(np.divide(state,1000)).float())
      return 1.0-pred

if __name__ == "__main__":
  board = chess.Board()
  neuralValuator = NeuralValuator()
  print(neuralValuator.NeuralValue(board))