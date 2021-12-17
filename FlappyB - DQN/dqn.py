from flappyBird import draw_floor, create_pipe, move_pipes, draw_pipes, check_collision, flap_step, resets
import pygame, sys
import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import time
import os

#Pygame
pygame.init()
screen = pygame.display.set_mode((576, 1024))

#Game Variables
gravity = 0.25
bird_movement = 0

pipe_list = []
pipe_height = [400, 600, 800]

bg_surface = pygame.image.load('assets/background-day.png').convert() 
bg_surface = pygame.transform.scale2x(bg_surface)
floor_surface = pygame.image.load('assets/base.png').convert()
floor_surface = pygame.transform.scale2x(floor_surface)
floor_x_pos = 0

bird_surface = pygame.transform.scale2x(pygame.image.load('assets/bluebird-midflap.png')).convert_alpha()

pipe_surface = pygame.image.load('assets/pipe-green.png')
pipe_surface = pygame.transform.scale2x(pipe_surface)

SPAWNPIPE = pygame.USEREVENT #event for pipes spawning
pygame.time.set_timer(SPAWNPIPE, 1200) #triggers event every 1.2 seconds

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup = "flappyBird_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.batch_size = 16
        self.epsilon = 1
        self.epsilon_dec = 0.997
        self.epsilon_min = 0.01
        self.main_model = self.build_model(self.state_size, self.action_size)
    
    def build_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(24, input_shape = (self.state_size,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.epsilon = self.epsilon_min
        return model

    def save_model(self):
        self.main_model.save(self.weight_backup)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        prediction = self.main_model.predict(state)
        
        return np.argmax(prediction[0])

    def save_model(self):
        self.main_model.save(self.weight_backup)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        
    def replay(self,):
        if len(self.memory) < self.batch_size:
            return
        
        sample_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([step[0] for step in sample_batch])
        
        actions = np.asarray([step[1] for step in sample_batch])
        rewards = np.asarray([step[2] for step in sample_batch])
        next_states = np.asarray([step[3] for step in sample_batch])
        dones = np.asarray([step[4] for step in sample_batch])
        targets = rewards
        
        q_eval = self.main_model.predict(states)
        q_next = self.main_model.predict(next_states)
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next, axis=1)*dones
        _ = self.main_model.fit(states, q_target, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec


class flappyBird():

    def __init__(self):

        self.sample_batch_size = 100
        self.episodes = 500
        self.state_size = 3
        self.action_list = np.ndarray([0, 1], dtype=np.int8)
        self.action_size = 2
        self.agent = Agent(self.state_size, self.action_size)

    def run(self, bird_movement, pipe_list):
        
        clock = pygame.time.Clock()
        
        for episode in range(500):

            start = time.time()
            bird_rect = bird_surface.get_rect(center=(100, 512))
            pipe_list = []
            bird_movement = 0
            score = 0
            game_active = True

            state = np.asarray(resets(bird_movement, bird_rect)) #need to make this function

            while game_active:  

                for event in pygame.event.get():
                
                    #SpawnPipe
                    if event.type == SPAWNPIPE:

                        #Clearing pipeslist to get next pipe info
                        pipe_list = []
                        pipe_list.extend(create_pipe(pipe_height, pipe_surface)) 

                    #Quit
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    #Flap
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            #bird flaps up
                            bird_movement = 0
                            bird_movement -= 10

                #bliting
                #if episode % 20 == 0:
                #    screen.blit(bg_surface, (0, 0 ))
                #    screen.blit(bird_surface, bird_rect)
                #    draw_pipes(pipe_list, pipe_surface, screen)
                #    draw_floor(floor_surface, floor_x_pos, screen)  
                #    pygame.display.update()

                #main cycle
                action = self.agent.act(state) #gets info for step 
                
                if action == 0:
                    pass
                if action == 1:
                    bird_movement = 0
                    bird_movement -= 10
                new_state, reward, done = flap_step(action, pipe_list, bird_movement, bird_rect, self.state_size) #flap step returns state info
         
            
                game_active = done
                self.agent.remember(state, action, reward, new_state, int(done))
                state = new_state

                #Bird Movement
                bird_movement += gravity
                bird_rect.centery += bird_movement

                #Pipes Movement and Collision
                pipe_list = move_pipes(pipe_list)
                
                #more game variables
                score += 1
                clock.tick(120)

            
            score -= 1000
            self.agent.replay()

            print(f"Score: {score} - Episode: {episode} - Epsilon: {round(self.agent.epsilon, 3)} - Time: {round(time.time() - start, 3)}")

        self.agent.save_model()

flappy = flappyBird()
 
flappy.run(bird_movement, pipe_list)
