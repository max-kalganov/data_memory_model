import tensorflow as tf

if __name__ == '__main__':
    import numpy as np


    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.99)


    init_kernel = np.array([[1, -1]], dtype=np.float32)
    init_kernel2 = np.array([[1], [1]], dtype=np.float32)

    kernel_initializer = tf.keras.initializers.constant(init_kernel)
    kernel_initializer2 = tf.keras.initializers.constant(init_kernel2)
    tf.random.set_seed(1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, activation='relu', use_bias=False),
        tf.keras.layers.Dense(1, use_bias=False)
    ])

    rms = tf.keras.optimizers.SGD(lr=0.0001, clipvalue=0.5)
    model.compile(loss=tf.keras.losses.MAE, optimizer=rms)
    x = np.random.randint(-10000, 10000, 12800)
    y = np.abs(x)

    model.build((None, 1))
    print(model.weights)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(x, y, epochs=200, batch_size=128)

    x_test = np.random.randint(-10000, 10000, 128)
    y_test = np.abs(x_test)

    model.evaluate(x_test, y_test)
    print(*zip(x_test, model.predict(x_test)))
    print(model.weights)
