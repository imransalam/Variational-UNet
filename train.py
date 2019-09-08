
import tensorflow as tf
# tf.enable_eager_execution()

from unet import VariationalUnet
from data import preprocessing

def loss(y, y_bar, mus_sigmas):
    ls = 0
    for mu, sigma in mus_sigmas:
        ls = ls + (- 0.5 * tf.reduce_sum(1 + sigma - tf.math.pow(mu, 2) - tf.exp(sigma)))
    return tf.losses.mean_squared_error(y, y_bar) + ls

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction, mus_sigmas = model(inputs)
        loss_value = loss(mask, reconstruction, mus_sigmas)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction, mus_sigmas

def train():
    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    seed = 1
    model = VariationalUnet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    global_step = tf.Variable(0)

    num_epochs = 50
    batch_size = 1

    image_generator = image_datagen.flow_from_directory(
        'data/membrane/train/images/',
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/membrane/train/masks/',
        class_mode=None,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for (img,mask) in train_generator:
            img, mask = preprocessing(img, mask)
            # img = tf.expand_dims(img, axis=0)
            # mask = tf.expand_dims(mask, axis=0)
            
            loss_value, grads, reconstruction, mu, sigma = grad(model, img, mask)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
            
            if global_step.numpy() % 200 == 0:
                print("Step: {},         Loss: {}".format(global_step.numpy(),
                                              loss(x_inp, reconstruction, mus_sigmas).numpy()))


if __name__ == "__main__":
    train()