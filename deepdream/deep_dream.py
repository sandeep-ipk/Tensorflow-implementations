import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import inception5h


def load_img(filename):
    img = PIL.Image.open(filename)

    return np.float32(img)


def save_img(image, filename):
    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(img).save(file, 'jpeg')


def plot_img(image):

    if False:
        image = np.clip(image / 255.0, 0.0, 1.0)
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
        im = (PIL.Image.fromarray(image))
        im.show()


def normalize_img(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def resize_img(image, size=None, factor=None):
    if factor is not None:
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)
    else:
        size = size[0:2]
    size = tuple(reversed(size))
    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = PIL.Image.fromarray(img)
    img_resized = img.resize(size, PIL.Image.LANCZOS)
    img_resized = np.float32(img_resized)

    return img_resized


def get_tile_size(num_pixels, tile_size=400):

    num_tiles = int(round(num_pixels / tile_size))
    num_tiles = max(1, num_tiles)
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size

model = inception5h.Inception5h()

# this method's idea is taken from internet, to minimize memory usage while running this program
# it takes tiles/patches from image and computes gradients one tile at a time.

def gradient_tiles(gradient, image, tile_size=400):
    grad = np.zeros_like(image)
    x_max, y_max, _ = image.shape
    #  print("image: ", image.shape) #  rgb type, no confusion
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    x_tile_size4 = x_tile_size // 4
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4

    x_start = random.randint(-3 * x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        x_end = x_start + x_tile_size
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)
        y_start = random.randint(-3 * y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            y_end = y_start + y_tile_size
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)
            img_tile = image[x_start_lim:x_end_lim,
                       y_start_lim:y_end_lim, :]
            feed_dict = model.create_feed_dict(image=img_tile)
            g = session.run(gradient, feed_dict=feed_dict)
            g /= (np.std(g) + 1e-8)
            #  print("g: ", g.shape) #  shape same as tiled grad, read the optimize and tiled grad carefully.
            grad[x_start_lim:x_end_lim,
            y_start_lim:y_end_lim, :] = g
            y_start = y_end
        x_start = x_end

    return grad


def optimize(layer_tensor, image, num_iterations=10, step_size=3.0, tile_size=400,recursive_level=0):

    img1 = image.copy()
    gradient = model.get_gradient(layer_tensor)
    #print("gradient: ", gradient.shape)

    for i in tqdm(range(num_iterations)):
        grad = gradient_tiles(gradient=gradient, image=img1)
        sigma = (i * 4.0) / num_iterations + 0.5

        # can try this without the smoothing too
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        img1 += grad * step_size_scaled

    if recursive_level == 4:
        #print()
        #print("Image after:")
        plot_img(img1)

    return img1

# this recursive idea is not mine, but it helped to produce richer patterns.

def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):

    if num_repeats > 0:
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))
        img_downscaled = resize_img(image=img_blur,
                                      factor=rescale_factor)

        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats - 1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        img_upscaled = resize_img(image=img_result, size=image.shape)
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)
    img_result = optimize(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size, recursive_level=num_repeats)

    return img_result


session = tf.InteractiveSession(graph=model.graph)
image = load_img(filename='images/style4.jpg')
plot_img(image)

layer_tensor = model.layer_tensors[5][:,:,:,0:3]


'''
img_result = optimize(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=10,
                                step_size=3.0,
                                tile_size=400, recursive_level=4)

'''


img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)

save_img(img_result, 'lolz_5_0_3.jpeg')
