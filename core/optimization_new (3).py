import os
import numpy as np
import pyomo.environ as pyo
import gurobipy as gp
import omlt
import omlt.io
import omlt.neuralnet
import tensorflow as tf
from tensorflow import keras
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
tf.config.run_functions_eagerly(True)
import time

##################################################
from contextlib import redirect_stdout
import io
def read_model(filename):
    with io.StringIO() as buf, redirect_stdout(buf):
        return gp.read(filename)
##################################################


class ViolationEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold=1e-5):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        violation = logs.get('violation')
        
        if violation <= self.threshold:
            self.model.stop_training = True

# Algorithm 1 in paper Appendix
class Generator:
    def __init__(self, path,
                 binary_split, l, u, d_lower, d_upper, e_lower, e_upper, 
                 formulation=omlt.neuralnet.nn_formulation.ReluBigMFormulation, M=1000, m=1e-5, 
                 timelimit=None, solvelimit=1):
        self.path = path

        self.input_dim = len(l)
        self.binary_split = binary_split
        # l, u are kept solely as input bounds for OMLT's internal ReLU Big-M
        # propagation. They are NOT used as active safe-region constraints.
        self.l = l
        self.u = u
        self.d_lower = d_lower
        self.d_upper = d_upper
        self.e_lower = e_lower
        self.e_upper = e_upper
        self.formulation = formulation
        self.M = M
        self.m = m
        self.timelimit = timelimit
        self.solvelimit = solvelimit
        self.input_bounds = {i: (l[i], u[i]) for i in range(len(l))}

        self.phases = []
        self.modelcounts = []
        self.timelimits = []
        self.primals = []
        self.duals = []
        self.times = []
        self.measured_times = []
        self.terminations = []

        self.modelcount = 0

    def generate(self, h_base, h_delta, g, phase):
        try:
            hidden_dim = g.get_weights()[0].shape[0] 
            dummy_g = False
        except:
            hidden_dim = 1
            dummy_g = True

        def input_bounds(model, i):
            return (self.l[i], self.u[i])

        self.model = pyo.ConcreteModel()
        self.model.x1 = pyo.Var(range(self.input_dim), bounds=input_bounds)
        self.model.x2 = pyo.Var(range(self.input_dim), bounds=input_bounds)
        for i in range(self.binary_split, self.input_dim):
            self.model.x1[i].domain = pyo.Binary
            self.model.x2[i].domain = pyo.Binary
        self.model.z1 = pyo.Var(range(hidden_dim))
        self.model.z1_lower = pyo.Var(range(hidden_dim))
        self.model.z1_upper = pyo.Var(range(hidden_dim))
        self.model.z1_base = pyo.Var(range(hidden_dim))
        self.model.z1_delta = pyo.Var(range(hidden_dim), bounds=(0, None))  # delta >= 0
        self.model.z2 = pyo.Var(range(hidden_dim))
        self.model.z2_lower = pyo.Var(range(hidden_dim))
        self.model.z2_upper = pyo.Var(range(hidden_dim))
        self.model.z2_base = pyo.Var(range(hidden_dim))
        self.model.z2_delta = pyo.Var(range(hidden_dim), bounds=(0, None))  # delta >= 0
        self.model.y1 = pyo.Var()
        self.model.y2 = pyo.Var()
        self.model.gamma = pyo.Var(bounds=(self.m, None))
        self.model.b = pyo.Var(domain=pyo.Binary)
        # NOTE: t1, t2 binaries removed — clipping now encoded as direct
        # interval constraints [z_lower, z_upper], eliminating all Big-M
        # degeneracy from the auxiliary sections.

        @self.model.Constraint(range(self.input_dim))
        def property_lower(model, i):
            return model.x1[i] - model.x2[i] >= self.d_lower[i]
        @self.model.Constraint(range(self.input_dim))
        def property_upper(model, i):
            return model.x1[i] - model.x2[i] <= self.d_upper[i]
        self.model.violation_right = pyo.Constraint(expr=self.model.gamma <= self.model.y1 - self.model.y2 - self.e_upper + self.M*self.model.b)
        self.model.violation_left = pyo.Constraint(expr=self.model.gamma <= -self.model.y1 + self.model.y2 + self.e_lower + self.M*(1 - self.model.b))

        self.model.objective = pyo.Objective(expr=self.model.gamma, sense=pyo.maximize)

        ##################################################### Auxiliary 1 #####################################################
        # h_base encodes the lower bound directly: z1_lower = h_base(x1)
        self.model.h1_base = omlt.OmltBlock()
        formulation = self.formulation(omlt.io.load_keras_sequential(nn=h_base, scaled_input_bounds=self.input_bounds))
        self.model.h1_base.build_formulation(formulation)
        @self.model.Constraint(range(self.input_dim))
        def h1_base_input(model, i):
            return model.x1[i] == model.h1_base.inputs[i]
        @self.model.Constraint(range(hidden_dim))
        def h1_base_output(model, i):
            return model.z1_base[i] == model.h1_base.outputs[i]

        # h_delta encodes the non-negative width: z1_upper = z1_base + h_delta(x1)
        # h_delta uses softplus output so its values are strictly positive — no
        # consistency check needed, h_lower <= h_upper is guaranteed by construction.
        self.model.h1_delta = omlt.OmltBlock()
        formulation = self.formulation(omlt.io.load_keras_sequential(nn=h_delta, scaled_input_bounds=self.input_bounds))
        self.model.h1_delta.build_formulation(formulation)
        @self.model.Constraint(range(self.input_dim))
        def h1_delta_input(model, i):
            return model.x1[i] == model.h1_delta.inputs[i]
        @self.model.Constraint(range(hidden_dim))
        def h1_delta_output(model, i):
            return model.z1_delta[i] == model.h1_delta.outputs[i]

        # Derive lower/upper from base+delta (linear, no Big-M)
        @self.model.Constraint(range(hidden_dim))
        def z1_lower_def(model, i):
            return model.z1_lower[i] == model.z1_base[i]
        @self.model.Constraint(range(hidden_dim))
        def z1_upper_def(model, i):
            return model.z1_upper[i] == model.z1_base[i] + model.z1_delta[i]

        # Clipping: z1 ∈ [z1_lower, z1_upper] — two linear constraints, zero Big-M.
        # The LP solver pushes z1 adversarially within this interval; no binary
        # t1 variable needed. I1_lower is recovered post-solve from z1 ≈ z1_lower.
        @self.model.Constraint(range(hidden_dim))
        def clip_1_lower(model, i):
            return model.z1[i] >= model.z1_lower[i]
        @self.model.Constraint(range(hidden_dim))
        def clip_1_upper(model, i):
            return model.z1[i] <= model.z1_upper[i]

        ##################################################### Auxiliary 2 #####################################################
        # Same base+delta encoding for x2.
        self.model.h2_base = omlt.OmltBlock()
        formulation = self.formulation(omlt.io.load_keras_sequential(nn=h_base, scaled_input_bounds=self.input_bounds))
        self.model.h2_base.build_formulation(formulation)
        @self.model.Constraint(range(self.input_dim))
        def h2_base_input(model, i):
            return model.x2[i] == model.h2_base.inputs[i]
        @self.model.Constraint(range(hidden_dim))
        def h2_base_output(model, i):
            return model.z2_base[i] == model.h2_base.outputs[i]

        self.model.h2_delta = omlt.OmltBlock()
        formulation = self.formulation(omlt.io.load_keras_sequential(nn=h_delta, scaled_input_bounds=self.input_bounds))
        self.model.h2_delta.build_formulation(formulation)
        @self.model.Constraint(range(self.input_dim))
        def h2_delta_input(model, i):
            return model.x2[i] == model.h2_delta.inputs[i]
        @self.model.Constraint(range(hidden_dim))
        def h2_delta_output(model, i):
            return model.z2_delta[i] == model.h2_delta.outputs[i]

        # Derive lower/upper from base+delta (linear, no Big-M)
        @self.model.Constraint(range(hidden_dim))
        def z2_lower_def(model, i):
            return model.z2_lower[i] == model.z2_base[i]
        @self.model.Constraint(range(hidden_dim))
        def z2_upper_def(model, i):
            return model.z2_upper[i] == model.z2_base[i] + model.z2_delta[i]

        # Clipping: z2 ∈ [z2_lower, z2_upper]
        @self.model.Constraint(range(hidden_dim))
        def clip_2_lower(model, i):
            return model.z2[i] >= model.z2_lower[i]
        @self.model.Constraint(range(hidden_dim))
        def clip_2_upper(model, i):
            return model.z2[i] <= model.z2_upper[i]
        
        if dummy_g:
            self.model.output1 = pyo.Constraint(expr=self.model.y1 == self.model.z1[0])
            self.model.output2 = pyo.Constraint(expr=self.model.y2 == self.model.z2[0])

        else:
            ##################################################### g1 #####################################################
            self.model.g1 = omlt.OmltBlock()
            formulation = self.formulation(omlt.io.load_keras_sequential(g))
            self.model.g1.build_formulation(formulation)
            @self.model.Constraint(range(hidden_dim))
            def g1_input(model, i):
                return model.z1[i] == model.g1.inputs[i]
            self.model.g1_output = pyo.Constraint(expr=self.model.y1 == self.model.g1.outputs[0])
            
            ##################################################### g2 #####################################################
            self.model.g2 = omlt.OmltBlock()
            formulation = self.formulation(omlt.io.load_keras_sequential(g))
            self.model.g2.build_formulation(formulation)
            @self.model.Constraint(range(hidden_dim))
            def g2_input(model, i):
                return model.z2[i] == model.g2.inputs[i]
            self.model.g2_output = pyo.Constraint(expr=self.model.y2 == self.model.g2.outputs[0])

        ##################################################### Model to Solve #####################################################
        self.model.write(f'{self.path}/{self.modelcount}.lp', io_options={"symbolic_solver_labels": True})
        self.solver = read_model(f'{self.path}/{self.modelcount}.lp')

        ##################################################### Solve #####################################################
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
            self.solver.Params.OutputFlag = 1
#           self.solver.Params.LogFile = f'{self.path}/{self.modelcount}.log'
            self.solver.Params.LogToConsole = 0

        dump_files = False
        solve = 0
        self.solver.Params.TimeLimit = self.timelimit
        while (self.solver.SolCount == 0) and solve < self.solvelimit:
#           with open(f'{self.path}/{self.modelcount}.log', 'a') as logfile:
#               logfile.write("\n####################################################################################################\n")
#               logfile.write(f"solve: {solve}\n")
#               logfile.write("####################################################################################################\n")

            self.solver.Params.Seed = 0
            measured_time = time.time()
            self.solver.optimize()
            measured_time = time.time() - measured_time

#           if measured_time >= 1.5*self.solver.Params.TimeLimit:
#               dump_files = True

            self.phases.append(phase)
            self.modelcounts.append(self.modelcount)        
            self.timelimits.append(self.solver.Params.TimeLimit)
            self.times.append(self.solver.Runtime)
            self.measured_times.append(measured_time)
            self.terminations.append(self.solver.Status)        
            self.duals.append(self.solver.ObjBound)
            if self.solver.Status in [3,4]:
                self.primals.append(None)
                break
            else:
                self.primals.append(self.solver.ObjVal)

            self.solver.Params.TimeLimit = 2*self.solver.Params.TimeLimit
            solve += 1 

        if self.solver.SolCount > 0:
            x1 = np.array([self.solver.getVarByName(f'x1({i})').X for i in range(self.input_dim)])
            x2 = np.array([self.solver.getVarByName(f'x2({i})').X for i in range(self.input_dim)])
            z1_lower = np.array([self.solver.getVarByName(f'z1_lower({i})').X for i in range(hidden_dim)])
            z2_lower = np.array([self.solver.getVarByName(f'z2_lower({i})').X for i in range(hidden_dim)])
            z1 = np.array([self.solver.getVarByName(f"z1({i})").X for i in range(hidden_dim)])
            z2 = np.array([self.solver.getVarByName(f"z2({i})").X for i in range(hidden_dim)])
            # I_lower: whether the adversarial z was pushed to its lower bound.
            # Previously encoded by t1/t2 binaries; now read directly from the
            # continuous z values — semantics are identical.
            I1_lower = np.abs(z1 - z1_lower) <= 10**-5
            I2_lower = np.abs(z2 - z2_lower) <= 10**-5
            counter1 = [np.array([x1]), np.array([I1_lower])]
            counter2 = [np.array([x2]), np.array([I2_lower])]
        else:
            counter1 = None
            counter2 = None

        if not dump_files:
#           os.system(f'rm {self.path}/{self.modelcount}.log')
            # os.system(f'rm {self.path}/{self.modelcount}.lp')
            os.remove(f'{self.path}/{self.modelcount}.lp')
        self.modelcount += 1

        return counter1, counter2


# Algorithm 4 in main paper
class Projector(keras.Model):
    def __init__(self, h_base, h_delta, g, e_lower, e_upper, n_counter):
        super().__init__()
        self.h_base = h_base
        self.h_delta = h_delta
        self.g = g
        
        self.e_lower = e_lower
        self.e_upper = e_upper

        self.theta_h_base = self.h_base.get_weights()
        self.theta_h_delta = self.h_delta.get_weights()
        self.theta_g = self.g.get_weights()

        self.lambda_ = tf.Variable(tf.zeros(shape=(n_counter,)), name='lambda', trainable=True)

        self.dual_optimizer = keras.optimizers.Adam()

        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.deviation_tracker = keras.metrics.Mean(name='deviation')
        self.penalization_tracker = keras.metrics.Mean(name='penalization')
        self.violation_tracker = keras.metrics.Mean(name='violation')

    def call(self, inputs):
        x, I_lower = inputs
        h_base_out = self.h_base(x)
        h_upper_out = h_base_out + self.h_delta(x)
        # h_lower_out = h_base_out — ordering guaranteed by construction
        z = tf.where(I_lower, h_base_out, h_upper_out)
        y = self.g(z)
        
        return y

    def _individual_deviation(self, model, theta_ref):
        theta = model.trainable_variables

        loss = 0
        for theta_var, theta_ref_var in zip(theta, theta_ref):
            loss += tf.math.reduce_mean(tf.math.square(theta_var - theta_ref_var))

        return loss

    def _deviation(self):
        loss = 0
        loss += self._individual_deviation(self.h_base, self.theta_h_base)
        loss += self._individual_deviation(self.h_delta, self.theta_h_delta)
        loss += self._individual_deviation(self.g, self.theta_g) 

        return loss

    def _individual_violation(self, e, y1, y2):
        loss = tf.math.maximum(0, y1 - y2 - e)

        return loss

    def _custom_loss(self, input1, input2):
        y1, y2 = self(input1, training=True), self(input2, training=True)

        deviation = self._deviation()

        violation_lower = self._individual_violation(-self.e_lower, y2, y1)
        violation_upper = self._individual_violation(self.e_upper, y1, y2)
        violation = violation_lower + violation_upper

        penalization = tf.reduce_sum(tf.multiply(self.lambda_, violation))

        loss = deviation + penalization
         
        return loss, deviation, penalization, violation

    def train_step(self, data):
        input1, input2 = data

        # Descent Step
        with tf.GradientTape() as tape:
            loss, deviation, penalization, violation = self._custom_loss(input1, input2)

        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        # Ascent Step
        with tf.GradientTape() as tape:
            loss, deviation, penalization, violation = self._custom_loss(input1, input2)
            loss = -loss

        trainable_variables = [self.lambda_]
        grads = tape.gradient(loss, trainable_variables)
        self.dual_optimizer.apply_gradients(zip(grads, trainable_variables))
        loss = -loss

        self.loss_tracker.update_state(loss)
        self.deviation_tracker.update_state(deviation)
        self.penalization_tracker.update_state(penalization)
        self.violation_tracker.update_state(violation)
        
        return {'loss': self.loss_tracker.result(), 
                'deviation': self.deviation_tracker.result(),
                'penalization': self.penalization_tracker.result(),
                'violation': self.violation_tracker.result()
                }