import gym
import numpy as np

# in this part, we will get the part of cartPole from the whole screen as
# the input for our model

def get_screen(env):
    # get the whole screen with the shape (height, width, rgb_channel)
    screen = env.render("rgb_array")
    screen = screen.mean(axis=2) # rgb to gray
   
    screen_height, screen_width = screen.shape  
    screen = screen.reshape(screen_height, screen_width, 1)
    world_width = env.x_threshold * 2 
    scale = screen_width / world_width                  
    cart_width = 50
    cart_height = 30 
    pole_length = 2 * env.length
    size_need = cart_height + scale * pole_length
    # cart x position
    cartx = env.state[0] * scale + screen_width / 2 

    cartPole_screen = screen[int(318-size_need):318, int(cartx-size_need/2) : int(cartx+size_need/2)]
    
    # cropping to lower resolution with the shape (78, 78)
    return cartPole_screen[::4, ::4] / 255.0

def get_observation(env, mode=None):
    if mode != "raw_image":
        return env.state
    else:
        return get_screen(env)

if __name__ == '__main__':
    env = gym.make("CartPole-v0")

    env.reset()
    print(env.state.shape)


