import pygame, sys
import random
import numpy as np

def draw_floor(floor_surface, floor_x_pos, screen):
    screen.blit(floor_surface, (floor_x_pos, 900))
    screen.blit(floor_surface, (floor_x_pos + 576, 900))

def create_pipe(pipe_height, pipe_surface):
    random_pipe_pos = random.choice(pipe_height)
    bottom_pipe = pipe_surface.get_rect(midtop = (700, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom = (700, random_pipe_pos - 300))
    return bottom_pipe, top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 5
    return pipes

def draw_pipes(pipes, pipe_surface, screen):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface, pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe)

def check_collision(pipes, bird_rect):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return False

    if bird_rect.top <= 0 or bird_rect.bottom >= 900:
        return False

    return True

def resets(bird_movement, bird_rect):
    #return list of info for flappy bird distance from pipes etc evven though pipe list is nothing

    return np.asarray([100, bird_rect.centery - 450, bird_movement])

def flap_step(step, pipe_list, bird_movement, bird_rect, state_size):
    
    #check collision is here for the done flag
    game_active = check_collision(pipe_list, bird_rect)
     
    if game_active == False:
        reward = -1000
    else: 
        reward = 0

    if pipe_list == []:
        #will have to change to differene from 450

        stateInfo = [np.asarray([100, bird_rect.centery - 450, bird_movement]), reward, game_active]
        return stateInfo

    differenceY = bird_rect.centery - pipe_list[0][1] - 150
    differenceX = bird_rect.centerx - pipe_list[0][0]
    
    state = np.asarray([differenceX, differenceY, bird_movement])
    stateInfo = [state, reward, game_active]
    
    #may want to change to int64 later and nd array
    return stateInfo