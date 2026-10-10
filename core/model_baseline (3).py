import sys
import tensorflow as tf
from tensorflow import keras
from core.optimization_baseline import *
tf.config.run_functions_eagerly(True)


class TorchLikeInitializer(keras.initializers.Initializer):
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        bound = 1/np.sqrt(np.prod(shape[:-1]))
        return tf.random.uniform(shape, -bound, bound, dtype, self.seed)

    def get_config(self):
      return {'seed': self.seed}
 

class BaseNet(keras.Model):
    def __init__(self, architecture, model='smle', **kwargs):
        super().__init__(**kwargs)
        self._layers = []

        for layer in architecture:
            size, _type = layer
            if _type == 'conv':
                if model == 'crownibp':
                    self._layers.append(keras.layers.ZeroPadding2D(padding=(1, 1)))
                    self._layers.append(keras.layers.Conv2D(filters=size, kernel_size=(3,3), padding='valid', activation='relu', strides=(2, 2),
                                                            kernel_initializer=TorchLikeInitializer(), bias_initializer=TorchLikeInitializer()))
                else:
                    self._layers.append(keras.layers.Conv2D(filters=size, kernel_size=(3,3), padding='same', activation='relu', strides=(2, 2),
                                                            kernel_initializer=TorchLikeInitializer(), bias_initializer=TorchLikeInitializer()))
            elif _type == 'flatten':
                self._layers.append(keras.layers.Flatten())
            else:
                self._layers.append(keras.layers.Dense(units=size, activation=_type, 
                                                       kernel_initializer=TorchLikeInitializer(), bias_initializer=TorchLikeInitializer()))
                                 
    def call(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs


# SMiLE Architecture
class Net(keras.Model):
    def __init__(self, model_path, mode,
                 h, g, h_lower, h_upper, 
                 binary_split, l, u, d_lower, d_upper, e_lower, e_upper, 
                 pretrain, generation_timelimit, generation_solvelimit, generation_frequency, 
                 lambda_box_pretrain=1.5, lambda_box_train=1.5, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.mode = mode

        self.h = h
        self.h_lower = h_lower
        self.h_upper = h_upper
        self.g = g
        self.flatten = keras.layers.Flatten()
        
        self.d_lower = d_lower
        self.d_upper = d_upper
        self.e_lower = e_lower
        self.e_upper = e_upper
        self.enforce_prop = (d_lower is not None) and (d_upper is not None) and (e_lower is not None) and (e_upper is not None)

        self.box = 0
        self.violation = 0
        self.penalization = 0
        self.max_violation = 10**-5
        self.pretrain = pretrain
        self.lambda_box_pretrain = lambda_box_pretrain
        self.lambda_box_train = lambda_box_train

        self.projection_iterationlimit = 100
        self.enforcement_iterationlimit = 50

        self.lambda_violation = 0
        if self.enforce_prop:
            self.lambda_violation = tf.Variable(tf.zeros(shape=(1,)), name='lambda', trainable=True)
            self.dual_optimizer = keras.optimizers.Adam()
            self.generator = Generator(path=model_path, 
                                       binary_split=binary_split, l=l, u=u, d_lower=d_lower, d_upper=d_upper, e_lower=e_lower, e_upper=e_upper,
                                       timelimit=generation_timelimit, solvelimit=generation_solvelimit)
            self.counterexamples = []
            self.generation_frequency = generation_frequency

        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.accuracy_tracker = keras.metrics.Mean(name='accuracy')
        self.box_tracker = keras.metrics.Mean(name='box')
        self.violation_tracker = keras.metrics.Mean(name='violation')
        self.penalization_tracker = keras.metrics.Mean(name='penalization')
        self.lambda_violation_tracker = keras.metrics.Mean(name='lambda_violation')

    # Algorithm 2 in paper Apppendix
    def call_counterexample(self, input_, training):
        x, I_lower = input_
        z = tf.where(I_lower, self.h_lower(x, training=training), self.h_upper(x, training=training))
        y = self.g(z, training=training)

        return y
    
    # Algorithm 3 in paper Apppendix
    def isvalid(self, counter1, counter2):
        if (counter1 is not None) and (counter2 is not None):
            y1 = self.call_counterexample(counter1, training=False)
            y2 = self.call_counterexample(counter2, training=False)
            violation = self._compute_violation(y1, y2)
            isvalid = violation > self.max_violation
        else:
            isvalid = False

        return isvalid

    def _compute_individual_violation(self, e, y1, y2):
        loss = tf.math.maximum(0, y1 - y2 - e)

        return loss

    def _compute_violation(self, y1, y2):
        violation_lower = self._compute_individual_violation(-self.e_lower, y2, y1)
        violation_upper = self._compute_individual_violation(self.e_upper, y1, y2)
        violation = violation_lower + violation_upper

        return violation
    
    # Algorithm 3 in main paper
    def _custom_loss(self, x, y, counter1, counter2):
        y_pred = self(x, training=True)
        accuracy = self.compute_loss(y=y, y_pred=y_pred)

        if (counter1 is not None) and (counter2 is not None):
            y1 = self.call_counterexample(counter1, training=True)
            y2 = self.call_counterexample(counter2, training=True)
            violation = self._compute_violation(y1, y2)
            penalization = tf.reduce_sum(tf.multiply(self.lambda_violation, violation))
        else:
            violation = tf.constant(0.)
            penalization = tf.reduce_sum(tf.multiply(self.lambda_violation, violation))

        z_pred_h = self.h(x, training=True)
        z_pred_h_lower = self.h_lower(self.flatten(x), training=True)
        z_pred_h_upper = self.h_upper(self.flatten(x), training=True)
        gap_upper = tf.reduce_mean(tf.math.maximum(0, z_pred_h - z_pred_h_upper))
        gap_lower = tf.reduce_mean(tf.math.maximum(0, z_pred_h_lower - z_pred_h))
        box = tf.multiply(self.lambda_box_train, (gap_upper + gap_lower))

        loss =  accuracy + penalization + box

        return loss, accuracy, box, violation, penalization

    # Algorithm 2 in main paper
    def _pretrain_loss(self, x, y):
        y_pred = self(x, training=True)
        y_pred_h = self.g(self.h(x, training=True))
        y_pred_h_lower = self.g(self.h_lower(self.flatten(x), training=True))
        y_pred_h_upper = self.g(self.h_upper(self.flatten(x), training=True))
        accuracy = self.compute_loss(y=y, y_pred=y_pred)
        accuracy_h = self.compute_loss(y=y, y_pred=y_pred_h)
        accuracy_h_lower = self.compute_loss(y=y, y_pred=y_pred_h_lower)
        accuracy_h_upper = self.compute_loss(y=y, y_pred=y_pred_h_upper)
        accuracy = accuracy + accuracy_h + accuracy_h_lower + accuracy_h_upper

        z_pred_h = self.h(x, training=True)
        z_pred_h_lower = self.h_lower(self.flatten(x), training=True)
        z_pred_h_upper = self.h_upper(self.flatten(x), training=True)
        gap_upper = tf.reduce_mean(tf.math.maximum(0, z_pred_h - z_pred_h_upper))
        gap_lower = tf.reduce_mean(tf.math.maximum(0, z_pred_h_lower - z_pred_h))
        box = tf.multiply(self.lambda_box_pretrain, (gap_upper + gap_lower))

        loss = accuracy + box

        return loss, accuracy, box

    # Algorithm 1 in main paper
    def train_step(self, data):
        x, y = data

        if self.optimizer.iterations.numpy() == 0:
            y_pred = self(x)

        if not self.enforce_prop:
            if self.pretrain:
                with tf.GradientTape() as tape:
                    loss, accuracy, self.box = self._pretrain_loss(x=x, y=y)
            else:
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compute_loss(y=y, y_pred=y_pred)
                    accuracy = loss

            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        else:
            if self.optimizer.iterations.numpy() % self.generation_frequency == 0:
                if len(self.counterexamples) == 0 or not self.isvalid(self.counterexamples[-1][0], self.counterexamples[-1][1]): 
                    counter1, counter2 = self.generator.generate(h_lower=self.h_lower, h_upper=self.h_upper, g=self.g, phase=1)
                    self.counterexamples.append((counter1, counter2))
            counter1, counter2 = self.counterexamples[-1]

            # Descent Step
            with tf.GradientTape() as tape:
                loss, accuracy, self.box, self.violation, self.penalization = self._custom_loss(x, y, counter1, counter2)

            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

            # Ascent Step
            with tf.GradientTape() as tape:
                loss, accuracy, self.box, self.violation, self.penalization = self._custom_loss(x, y, counter1, counter2)
                loss = -loss

            trainable_variables = [self.lambda_violation]
            gradients = tape.gradient(loss, trainable_variables)
            self.dual_optimizer.apply_gradients(zip(gradients, trainable_variables))
            loss = -loss

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        self.box_tracker.update_state(self.box)
        self.violation_tracker.update_state(self.violation)
        self.penalization_tracker.update_state(self.penalization)
        self.lambda_violation_tracker.update_state(self.lambda_violation)

        return {
                'loss': self.loss_tracker.result(),
                'accuracy': self.accuracy_tracker.result(),
                'box': self.box_tracker.result(),
                'violation': self.violation_tracker.result(),
                'penalization': self.penalization_tracker.result(),
                'lambda_violation': self.lambda_violation_tracker.result(),
                }

    def enforce(self, l, u, d_lower, d_upper, e_lower, e_upper):
        self.enforcement_iteration = 0

        #################### Verify (Initial) #########################
        counter1, counter2 = self.generator.generate(h_lower=self.h_lower, h_upper=self.h_upper, g=self.g, phase=2)
        self.counterexamples.append((counter1, counter2))
        if (counter1 is not None) and (counter2 is not None):
            self.violation = self.generator.solver.ObjVal
        else:
            self.violation = 0
        ###############################################################

        while self.violation > self.max_violation and self.enforcement_iteration < self.enforcement_iterationlimit:
            ######################## Enforce #############################
            counter1, counter2 = self.counterexamples[-1]
            violation_early_stopping = ViolationEarlyStopping(threshold=self.max_violation)
            projector = Projector(h_lower=self.h_lower, h_upper=self.h_upper, g=self.g, e_lower=e_lower, e_upper=e_upper, n_counter=1)
            projector.compile(optimizer='adam', loss='mse')
            projector.fit(counter1, counter2, epochs=self.projection_iterationlimit, batch_size=1, callbacks=[violation_early_stopping], verbose=0)

            #################### Verify #########################
            counter1, counter2 = self.generator.generate(h_lower=self.h_lower, h_upper=self.h_upper, g=self.g, phase=2)
            self.counterexamples.append((counter1, counter2))
            if (counter1 is not None) and (counter2 is not None):
                self.violation = self.generator.solver.ObjVal
            else:
                self.violation = 0
            ###############################################################

            ############################# Logging ##################################
            sys.stdout.write(f'\riteration: {self.enforcement_iteration}  --  violation: {self.violation}')
            sys.stdout.flush()
            ########################################################################

            self.enforcement_iteration += 1
        print('\n==============================================')

    def call(self, inputs):
        if self.h_lower:
            outputs = self.g(tf.maximum(tf.minimum(self.h(inputs), self.h_upper(self.flatten(inputs))), self.h_lower(self.flatten(inputs))))
        else:
            outputs = self.g(self.h(inputs))

        if self.mode == 'classification':
            outputs = tf.nn.sigmoid(outputs)

        return outputs