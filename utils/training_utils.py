import tensorflow as tf


def run_adam(model, num_iter, train_iter, lr, compile=True):
    training_loss = model.training_loss_closure(train_iter, compile=compile)
    optimizer = tf.optimizers.Adam(lr)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    print('{:>5s}'.format("iter") + '{:>24s}'.format("ELBO:"))
    iters = []
    elbos = []
    for i in range(1, num_iter + 1):
        try:
            optimization_step()

            if i % 5 == 0 or i == 0:
                elbo = -training_loss().numpy()
                print('{:>5d}'.format(i) + '{:>24.6f}'.format(elbo))
                iters.append(i)
                elbos.append(elbo)
        except KeyboardInterrupt as e:
            print("stopping training")
            break

    return iters, elbos
