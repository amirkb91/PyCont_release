import optax
import jax.numpy as jnp
import jax
import haiku as hk
import matplotlib.pyplot as plt
import os
import pickle

from haiku.initializers import VarianceScaling
from functools import partial
from datetime import datetime


class Damped_MLP(hk.Module):
    """
    Define the 1DOF damped lagrangian neural network with this class.
    Pass in the 'settings' dictionary as the MODEL variable.
    An additional dissipation function, self.damp_mlp, is constructed to model damping forces.
    """

    # Initialise the model
    def __init__(self, MODEL, name=None):
        super().__init__(name=MODEL['name'])
        lag_layers, damp_layers = [], []
        lag_units, damp_units = MODEL['lag_units'], MODEL['damp_units']
        layers = MODEL['layers']
        activation = jax.nn.softplus
        glorot_uniform = VarianceScaling(1.0, 'fan_avg', 'uniform')

        for layer in range(layers):
            lag_layers.append(
                hk.Linear(
                    output_size=lag_units,
                    w_init=glorot_uniform,
                    b_init=glorot_uniform
                )
            )

            damp_layers.append(
                hk.Linear(
                    output_size=damp_units,
                    w_init=glorot_uniform,
                    b_init=glorot_uniform
                )
            )

            lag_layers.append(activation)
            damp_layers.append(activation)

        lag_layers.extend([hk.Linear(
            output_size=1,
            w_init=glorot_uniform,
            b_init=glorot_uniform
        )
        ])

        damp_layers.extend([hk.Linear(
            output_size=1,
            w_init=glorot_uniform,
            b_init=glorot_uniform
        )
        ])

        self.lag_mlp = hk.Sequential(lag_layers)
        self.damp_mlp = hk.Sequential(damp_layers)

    # Apply the model
    def __call__(self, x):
        q, d_q = jnp.split(x, 2, -1)
        # Dissipation network only recieves velocity data here
        return self.lag_mlp(x), self.damp_mlp(d_q)


class BaseModel():
    """
    Set up the functions for training and testing as attributes of the model so that they can be called repetitively and easily.
    """

    def __init__(self, hk_module, optimizer, settings, info, phy_sys, seed=0):
        self.input_shape = settings['input_shape']
        self.hk_module = hk_module
        self.info = info
        self.phy_sys = phy_sys
        self.settings = settings
        self.optimizer = optimizer
        self.key = jax.random.PRNGKey(seed)

    def _compile(self, model_params, HK_Module, RNG, init_data, **kwargs):
        """Compile the network and initialize model parameters"""
        def forward_fn(x):
            module = HK_Module(model_params, **kwargs)
            return module(x)

        # Construct network
        net = hk.without_apply_rng(hk.transform(forward_fn))
        # Initialize weights & biases of network
        params = net.init(rng=RNG, x=init_data)

        return params, net

    def gather(self):
        """Initialize network parameters, the haiku function containing the network, the loss function && the update function"""
        # Initialize network parameters with the shape of init_data
        init_data = jnp.zeros((self.input_shape), dtype=jnp.float32)
        params, net = self._compile(
            self.settings, self.hk_module, self.key, init_data)

        # Assign to class attributes
        self.net = net
        self.init_params = params
        self.loss = self._loss()
        self.init_opt_state, self.update = self._update()

        return True

    def _loss(self):
        """MSE loss function"""
        @jax.jit
        def loss_fn(params, batch):
            X, Y = batch
            pred, _ = self.net.apply(params, None, X)
            return jnp.mean((jnp.square(pred - Y)))

        return loss_fn

    def _update(self):
        """Update function that will update network parameters"""
        loss_fn = self._loss()
        opt = self.optimizer
        opt_init_state = opt.init(self.init_params)

        @jax.jit
        def update(params, opt_state, batch):
            loss_value, grads = jax.value_and_grad(loss_fn, 0)(params, batch)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        return opt_init_state, update

    def init_step(self):
        """Initial step to get the initial optimizer state"""
        return self.init_params, self.init_opt_state

    def train_step(self, params, opt_state, batch):
        """Define each training step"""
        params, opt_state, loss_value = self.update(params, opt_state, batch)
        return params, opt_state, loss_value

    def test_step(self, params, batch):
        """Test step to gauge network's accuracy"""
        batch_loss = self.loss(params, batch)
        return batch_loss

    def plot_results(self, results):
        X = [step for step in range(results['last_epoch'])]
        Y1 = results['metrics']['train_loss']
        Y2 = results['metrics']['test_loss']
        plt.plot(X, Y1, label='Training Loss', color='blue')
        plt.plot(X, Y2, label='Test Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()

    def DataIterator(self, X=None, batch=-1, shuffle=False, seed=0):
        assert type(X) == tuple
        n = len(X[0])
        batches = int(jnp.ceil(n/batch))
        idx = jnp.arange(0, n)
        if shuffle:
            rng = jax.random.PRNGKey(seed)
            idx = jax.random.permutation(rng, idx)

        if batch < 0:
            yield tuple([x[idx] for x in X])
        else:
            for i in range(batches):
                start = i*batch
                index = idx[start:start+batch]
                result = tuple([x[index] for x in X])
                yield jax.device_put(result)

    def train(self, dataset, test_dataset, results=None, epochs=50, show_every=10):
        """dataset: training dataset (jnp.array),
           test_dataset: test_dataset (jnp.array),
           results: previous results if continuing training on a previously trained model (dict),
           epochs: number of desired training epochs (int),
           show_every: display training and test losses every n epochs (int)
        """
        train_batch_size = self.settings['train_batch_size']

        if self.settings['test_batch_size'] == -1:
            test_batch_size = len(test_dataset[0])
        else:
            test_batch_size = self.settings['test_batch_size']

        shuffle = self.settings['shuffle']
        seed = self.settings['seed']

        if results == None:
            results = {}

        _params, _opt_state = self.init_step()
        params = results.get('params', _params)

        _metrics = {'train_loss': [], 'test_loss': []}
        metrics = results.get('metrics', _metrics)

        _best_loss = jnp.inf
        best_loss = results.get('best_loss', _best_loss)

        best_params = results.get('best_params', None)

        opt_state = results.get('opt_state', _opt_state)

        _start_epoch = 0
        start_epoch = results.get('last_epoch', _start_epoch)

        start_time = datetime.now()

        for step in range(start_epoch, start_epoch+epochs):
            # ---------------------------------------------------------
            # train step
            train_epoch_loss = 0
            batches = 0
            train_batches = partial(
                self.DataIterator, batch=train_batch_size, shuffle=shuffle, seed=seed)

            for train_batch in train_batches(dataset):
                batches += 1
                params, opt_state, train_batch_loss = self.train_step(
                    params, opt_state, train_batch)
                train_epoch_loss += train_batch_loss

            train_epoch_loss /= (train_batch_size*batches)
            metrics['train_loss'].append(train_epoch_loss)

            # ---------------------------------------------------------
            # test step
            test_epoch_loss = 0
            test_batches_counter = 0
            test_batches = partial(
                self.DataIterator, batch=test_batch_size, shuffle=True, seed=50)

            for test_batch in test_batches(dataset):
                test_batches_counter += 1
                test_batch_loss = self.test_step(params, test_batch)
                test_epoch_loss += test_batch_loss

            test_epoch_loss /= (test_batch_size*test_batches_counter)
            metrics['test_loss'].append(test_epoch_loss)

            # ---------------------------------------------------------
            # check loss
            if test_epoch_loss < best_loss:
                best_loss = test_epoch_loss
                best_params = jax.device_get(params)

            # ---------------------------------------------------------
            # update the results dictionary
            if step % show_every == 0:

                print(
                    f'Epoch: {step} | Train Loss: {"%.8f" % train_epoch_loss} | Best Loss: {"%.8f" % best_loss} | Test Loss: {"%.8f" % test_epoch_loss}')
                print(
                    '--------------------------------------------------------------------------------------------------------')

        time_taken = datetime.now() - start_time

        self.best_params = jax.device_get(best_params)
        self.results = results

        results = {'metrics': jax.device_get(metrics),
                   'best_loss': jax.device_get(best_loss),
                   'best_params': jax.device_get(best_params),
                   'params': jax.device_get(params),
                   'opt_state': jax.device_get(opt_state),
                   'last_epoch': jax.device_get(step+1),
                   'settings': self.settings,
                   'time_taken': time_taken.total_seconds()/60,
                   'phy_sys': self.phy_sys,
                   'info': self.info}

        print(
            f'Final Epoch ---> Train Loss: {"%.8f" % train_epoch_loss} |Best Loss: {"%.8f" % best_loss} | Test Loss: {test_epoch_loss}')

        return results

    def save_model(self, results, DATA_DIR=''):
        os.makedirs(DATA_DIR, exist_ok=True)
        model_fn = DATA_DIR + '/' + 'model.pkl'
        metrics_fn = DATA_DIR + '/' + 'metrics.pkl'

        pickle.dump({'best_params': results['best_params'],
                    'best_loss': results['best_loss'],
                     'opt_state': results['opt_state'],
                     'settings': results['settings'],
                     'params': results['params'],
                     'last_epoch': results['last_epoch'],
                     'phy_sys': results['phy_sys'],
                     'info': results['info'],
                     'time_taken': results['time_taken']},
                    open(model_fn, 'wb'))

        pickle.dump(results['metrics'],
                    open(metrics_fn, 'wb'))

        return model_fn, metrics_fn

    @staticmethod
    def load_model(file_name):

        with open(file_name, 'rb') as file_cont:
            data = pickle.load(file_cont)

        return data



class Damped_LNN(BaseModel):

    def __init__(self, hk_module, optimizer, settings, info, phy_sys, seed=0):
        super().__init__(hk_module, optimizer, settings, info, phy_sys, seed)

    def _eom(self):

        def eom(L, x, f):
            q, q_t = jnp.split(x, 2, axis=-1)

            M = jax.hessian(L, 1)(q, q_t)[0]
            KQ = jax.jacobian(L, 0)(q, q_t)[0]
            CQ = jax.jacobian(L, 1)(q, q_t)[1]
            # ddL/dqdq_t term in Lagrangian formulation for eom
            GC1 = jax.jacfwd(jax.jacrev(L, 1), 0)(q, q_t)[0]
            GCQ = jnp.tensordot(GC1, q_t, axes=1)

            S = KQ + f - CQ - GCQ
            invM = jnp.linalg.pinv(M)
            q_tt1 = jnp.tensordot(S, invM, axes=1)
            # q_tt = jnp.expand_dims(q_tt1.squeeze(1), -1)

            return jnp.concatenate([q_t, q_tt1], axis=-1)

        return eom

    def _dynamics(self):
        net = self.net

        qmax = jnp.array([self.info['qmax']], dtype=jnp.float32)
        q_dmax = jnp.array([self.info['qdmax']], dtype=jnp.float32)

        def dynamics(params):

            def zeros(x):
                """Enforce boundary conditions"""
                z = jnp.zeros_like(x)
                # Lagrangian & Damping NN
                l, d = net.apply(params, z)
                dl, dd = jax.jacobian(net.apply, 1)(params, z)

                z1 = jnp.concatenate([l, d], axis=-1)
                dz1 = jnp.concatenate([dl, dd], axis=-1)

                return z1, dz1

            def lagrangian(qq, qq_t):
                """LNN forward step with normalized inputs"""
                assert qq.shape == (1,)

                qn = qq/qmax
                q_tn = qq_t/q_dmax

                state = jnp.concatenate([qn, q_tn], axis=-1)
                L, D = net.apply(params, state)

                return jnp.squeeze(L), jnp.squeeze(D)

            return lagrangian, zeros

        return dynamics

    def _loss(self):
        eom = self._eom()
        dynamics = self._dynamics()

        @jax.jit
        def loss_fn(params, batch):

            lagrangian, zeros = dynamics(params)
            q, f, q_d = batch

            # Split into disp. & vel.
            Q, Q_d = jnp.split(q, 2, axis=-1)

            # Calculate the right acceleration
            pred = jax.vmap(
                partial(eom, lagrangian))(q, f)
            z1, dz1 = jax.vmap(partial(zeros))(q)

            return jnp.mean(jnp.square(pred - q_d)) + jnp.mean(jnp.square(z1)) + jnp.mean(jnp.square(dz1))
        
        self.loss = loss_fn
        return self.loss

    def _predict(self, results):
        params = results['best_params']
        info = results['info']
        self.fmax = info['fmax']
        self.qmax = info['qmax']
        self.q_dmax = info['qdmax']

        qmax = self.qmax
        q_dmax = self.q_dmax

        eom = self._eom()
        dynamics = self._dynamics()
        lagrangian, _ = dynamics(params)

        @jax.jit
        def predict_accel(q, f):
            pred = jax.vmap(
                partial(eom, lagrangian))(q, f)
            return pred

        @jax.jit
        def predict_energy(q, q_t):
            qn = q/qmax
            q_tn = q_t/q_dmax

            state = jnp.concatenate(
                [jnp.expand_dims(qn, 0), jnp.expand_dims(q_tn, 0)], axis=-1)

            result = self.net.apply(params, state)

            return result

        self.predict_accel = predict_accel
        self.predict_energy = predict_energy

        return predict_accel, predict_energy
