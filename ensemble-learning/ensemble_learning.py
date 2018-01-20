import tensorflow as tf
import numpy as np
import os
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
total_images = np.concatenate([data.train.images, data.validation.images])
total_labels = np.concatenate([data.train.labels, data.validation.labels])

total_size = len(total_images)
train_size = int(0.8*total_size)


def random_training_set():
    idx = np.random.permutation(total_size)

    idx_train = idx[0:train_size]
    x_train = total_images[idx_train, :]
    y_train = total_labels[idx_train, :]


    return x_train, y_train


img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32, shape=[None,10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_wrap = pt.wrap(x_image)
norm = pt.BatchNormalizationArguments(scale_after_normalization=True)

with pt.defaults_scope(activation_fn= tf.nn.relu):
    y_pred, loss = x_wrap.\
        conv2d(kernel=5, depth=16, name='conv1', batch_normalize=norm).\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='conv2', batch_normalize=norm).\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)


session = tf.Session()


def init_variables():
    session.run(tf.global_variables_initializer())


train_batch_size = 64

# helper functions for predictions


def predict_labels(images):
    num_images = len(images)
    pred_labels = np.zeros(shape=(num_images, num_classes), dtype=np.float)
    i = 0

    while i<num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :]}
        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)
        i=j

    return pred_labels


def random_batch(x_train, y_train):
    num_images = len(x_train)
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)

    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]

    return x_batch, y_batch


def correct_prediction(images, labels, cls_true):
    pred_labels = predict_labels(images)
    cls_pred = np.argmax(pred_labels, axis=1)
    correct = (cls_true == cls_pred)

    return correct


def test_accuracy():
    correct = correct_prediction(images=data.test.images,
                              labels=data.test.labels,
                              cls_true=data.test.cls)
    return correct.mean()

# optimization:


def optimize(num_iterations, x_train, y_train):
    for i in tqdm(range(num_iterations)):
        x_batch, y_true_batch = random_batch(x_train, y_train)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i%100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "iteration: {0:>6}, training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))


num_networks = 5
num_iterations = 200


if True:
    for i in range(num_networks):
        print("Network: {0}".format(i+1))
        x_train, y_train = random_training_set()

        session.run(tf.global_variables_initializer())

        optimize(num_iterations=num_iterations, x_train=x_train, y_train=y_train)
        saver.save(sess=session, save_path=get_save_path(i))
        print()

batch_size = 128


def ensemble_predictions():
    pred_labels = []
    test_accuracies = []

    for i in range(num_networks):
        saver.restore(sess=session, save_path=get_save_path(i))
        test_acc = test_accuracy()
        test_accuracies.append(test_acc)

        msg = "Network: {0}, acc on Test-Set: {1:.4f}"
        print(msg.format(i+1, test_acc))

        pred = predict_labels(images = data.test.images)
        pred_labels.append(pred)

    return np.array(pred_labels), np.array(test_accuracies)


pred_labels, test_accuracies = ensemble_predictions()

print()
print("Mean test-set acc: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set acc:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set acc:  {0:.4f}".format(np.max(test_accuracies)))

#pred_labels.shape = (5, 10000, 10)

ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)

ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)
print("Ensemble test-set acc: {0:.4f}".format(np.mean(ensemble_correct)))
print()

best_net = np.argmax(test_accuracies)
#print(test_accuracies[best_net])

#print(ensemble_correct/len(ensemble_pred_labels))

best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)

total_correct_ensemble = np.sum(ensemble_correct)
total_correct_best_net = np.sum(best_net_correct)

print("total_correct_ensemble:", total_correct_ensemble, ", total_correct_best_net:", total_correct_best_net)



