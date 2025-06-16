import numpy as np
import random

class FruitCatacherEnv:
    def __init__(self, width=5, height=5): #preset values
        self.width = width #number of columns in the grid
        self.height = height #number of rows in the grid
        self.reset()

    def reset(self):
        self.basket_pos = self.width // 2 # Start basket in the center bottom
        self.fruit_pos = [random.randint(0, self.width - 1), 0] #Fruit appears at the top in a random column
        return self.get_state()

    def get_state(self):
        #Give a normalized state (0-1)
        return np.array ([
            self.basket_pos / self.width, #Basket column position
            self.fruit_pos[0] / self.width, #Fruit column position
            self.fruit_pos[1] / self.height # Fruit row position (0=top)
        ], dtype=np.float32)
    
    def step(self, action):
        if action == 0:
            self.basket_pos = max(0, self.basket_pos - 1) #move left unless at the edge
        elif action == 2:
            self.basket_pos = min(self.width - 1, self.basket_pos + 1) #move right unless at the edge
        
        self.fruit_pos[1] += 1 #fruit falls down one row

        #Check if fruit has reached the bottom row
        done = self.fruit_pos[1] == self.height
        distance = abs(self.fruit_pos[0] - self.basket_pos)
        reward = -distance / self.width  # Normalized distance penalty


        if done:
            if self.fruit_pos[0] == self.basket_pos:
                reward = 1 #fruit is caught, AI is rewarded
                print("Fruit caught!")
            else:
                reward = -1 #fruit is not caught
        

        return self.get_state(), reward, done
