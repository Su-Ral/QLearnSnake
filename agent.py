import torch
import numpy as np
import random
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_Qnet, QTrainer
from helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11,512,3)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval() 
    def get_state(self,game):
        head = game.snake[0]
        pointR = Point(head.x + 20,head.y)
        pointL = Point(head.x - 20,head.y)
        pointU = Point(head.x,head.y - 20)
        pointD = Point(head.x,head.y + 20)

        # pointR2 = Point(head.x + 40,head.y)
        # pointL2 = Point(head.x - 40,head.y)
        # pointU2 = Point(head.x,head.y - 40)
        # pointD2 = Point(head.x,head.y + 40)
    
    
        dirR = game.direction == Direction.RIGHT
        dirL = game.direction == Direction.LEFT
        dirU = game.direction == Direction.UP
        dirD = game.direction == Direction.DOWN

        state = [
            #danger straight
            (dirR and game.is_collision(pointR))or
            (dirU and game.is_collision(pointU))or
            (dirL and game.is_collision(pointL))or
            (dirD and game.is_collision(pointD)),
            #right
            (dirR and game.is_collision(pointD))or
            (dirU and game.is_collision(pointR))or
            (dirL and game.is_collision(pointU))or
            (dirD and game.is_collision(pointL)),
            #left
            (dirR and game.is_collision(pointU))or
            (dirU and game.is_collision(pointL))or
            (dirL and game.is_collision(pointD))or
            (dirD and game.is_collision(pointR)),

            # #danger straight
            # (dirR and game.is_collision(pointR2))or
            # (dirU and game.is_collision(pointU2))or
            # (dirL and game.is_collision(pointL2))or
            # (dirD and game.is_collision(pointD2)),
            # #right
            # (dirR and game.is_collision(pointD2))or
            # (dirU and game.is_collision(pointR2))or
            # (dirL and game.is_collision(pointU2))or
            # (dirD and game.is_collision(pointL2)),
            # #left
            # (dirR and game.is_collision(pointU2))or
            # (dirU and game.is_collision(pointL2))or
            # (dirL and game.is_collision(pointD2))or
            # (dirD and game.is_collision(pointR2)),

            dirL,dirR,dirU,dirD,
            #food left/right
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            #food up/down
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]  

        return np.array(state,dtype = int)
        
            
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            trainBatch = random.sample(self.memory,BATCH_SIZE)
        else:
             trainBatch = self.memory
        states,actions,rewards,next_states,dones = zip(*trainBatch)
        self.trainer.train(states,actions,rewards,next_states,dones)
            
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train(state,action,reward,next_state,done)
    def get_action(self,state,max_score):
        self.epsilon = 80 - (self.n_games*0.8 + max_score)
        finalMove = [0,0,0]
        if(self.n_games %2 == 0):
            print(self.epsilon)
        
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            finalMove[move] = 1
        else:
            ostate = torch.tensor(state, dtype=torch.float)
            predict = self.model(ostate)
            move = torch.argmax(predict).item()
            finalMove[move] = 1
        return finalMove

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state,record)
        reward,done,score = game.play_step(action)
        newState = agent.get_state(game)
        agent.train_short_memory(state,action,reward,newState,done)
        agent.remember(state,action,reward,newState,done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                #save agent model
            print('Game:', agent.n_games,'Score', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
        
if __name__ == '__main__':
    train() 
