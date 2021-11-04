import torch
import torchvision.transforms as T 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gym 

env = gym.make("CartPole-v1").unwrapped
env.reset()
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    # world_width means the width defined in the game, not the actual size of the game screen
    world_width = env.x_threshold * 2    
    scale = screen_width / world_width
    # return the cart position(middle of cart)
    # at the begining, the state of the env is none until the env is reset
    # the reason screen_width divided by 2 is the position defined in the game
    # using the axis
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    # env.render("rgb_array") will return a numpy array with shape (x, y, 3),  
    # representing RGB values for an x-by-y pixel image, suitable for turning into a video.
    # transposing the given array essentially means that replace the order of the axes to
    # make a new shape like below (transpose((2,0,1))) that will make the array be (3, x, y),
    # that is torch order (CHW). 
    screen = env.render("rgb_array").transpose((2, 0, 1))

    # height = 400, width = 600
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_heghit, screen_width = screen.shape
    screen = screen[:, int(screen_heghit*0.4):int(screen_heghit*0.8)]

    # only need view_width to view, not the whold width
    view_width = int(screen_width * 0.6)
    # cart_location will be the position in the screen image, not the env.state[0]
    cart_location = get_cart_location(screen_width)
    # choose the screen view according the cart_location, 
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # After strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]

    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # Resize, add a batch dimension
    return resize(screen).unsqueeze(0)

if __name__ == '__main__':
    
    plt.figure()
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    

    


