import tensorflow as tf
import numpy as np
import vgg16
import PIL.Image
from tqdm import tqdm


def load_img(filename, max_size=None):
    img = PIL.Image.open(filename)

    if max_size is not None:
        factor = max_size / np.max(img.size)
        size = np.array(img.size) * factor
        size = size.astype(int)
        img = img.resize(size, PIL.Image.LANCZOS)

    return np.float32(img)


def save_img(img, filename):
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(img).save(file, 'jpeg')


def plot_img(img):
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)

    im = (PIL.Image.fromarray(img))
    im.show()


def adam(grad, iter, m, v):
    beta1 = 0.9
    beta2 = 0.999
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m /= (1 - beta1 ** iter)
    v /= (1 - beta2 ** iter)
    return m, v


def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))


def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])

    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram


def content_loss(session, model, content_img, layer_ids):
    feed_dict = model.create_feed_dict(image=content_img)
    layers = model.get_layer_tensors(layer_ids)
    values = session.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            # make this value constant so as to not compute it again
            value_const = tf.constant(value)
            loss = mse(layer, value_const)
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def style_loss(session, model, style_img, layer_ids):
    feed_dict = model.create_feed_dict(image=style_img)
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]

        values = session.run(gram_layers, feed_dict=feed_dict)

        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            # make this value constant so as to not compute it again
            value_const = tf.constant(value)

            loss = mse(gram_layer, value_const)
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]))

    return loss


def optimize(content_img, style_img,
             content_layer_ids, style_layer_ids,
             weight_content=1.5, weight_style=10.0,
             weight_denoise=0.3,
             num_iterations=120, step_size=10.0):
    model = vgg16.VGG16()

    session = tf.InteractiveSession(graph=model.graph)

    loss_content = content_loss(session=session,
                                model=model,
                                content_img=content_img,
                                layer_ids=content_layer_ids)

    loss_style = style_loss(session=session,
                            model=model,
                            style_img=style_img,
                            layer_ids=style_layer_ids)

    loss_denoise = denoise_loss(model)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # we make the loss independent of the exact choice of style- and content-layers.
    loss_combined = weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    gradient = tf.gradients(loss_combined, model.input)
    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    # mixed img is initialized with random noise.
    mixed_img = np.random.rand(*content_img.shape) + 128
    # m = np.zeros(mixed_img.shape)
    # v = np.zeros(mixed_img.shape)

    for i in tqdm(range(num_iterations)):
        feed_dict = model.create_feed_dict(image=mixed_img)

        grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)
        grad = np.squeeze(grad)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the img by following the gradient.
        # m,v = adam(grad,i+1,m,v)
        # mixed_img -= step_size_scaled*m/(np.sqrt(v)+1e-8)

        mixed_img -= step_size_scaled * grad
        mixed_img = np.clip(mixed_img, 0.0, 255.0)

        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("iter:", i)

            msg = "weight adjustment for content: {0:.2e}, style: {1:.2e}, denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

    print()
    plot_img(mixed_img)
    session.close()
    return mixed_img


content_filename = 'images/style6.jpg'
content_img = load_img(content_filename, max_size=None)

style_filename = 'images/style5.jpg'
style_img = load_img(style_filename, max_size=300)

content_layer_ids = [4]

style_layer_ids = list(range(4))  # [0,1,2,3]

img = optimize(content_img=content_img,
               style_img=style_img,
               content_layer_ids=content_layer_ids,
               style_layer_ids=style_layer_ids,
               weight_content=1.5,
               weight_style=10.0,
               weight_denoise=0.3,
               num_iterations=30,
               step_size=10.0)

save_img(img, 's5_s8_A_1.jpeg')




