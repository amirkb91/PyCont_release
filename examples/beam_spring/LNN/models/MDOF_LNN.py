# Import ML packages
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from haiku.initializers import VarianceScaling

import matplotlib.pyplot as plt

from functools import partial
import os
import pickle
from datetime import datetime


class Physical_MLP(hk.Module):
    """
    Define a 1DOF MLP
    """
    # Initialise the model

    def __init__(self, settings, activation):
        super().__init__(name=settings["name"])
        n_layers = settings["layers"]
        layers = []
        units = settings["units"]
        activation = activation
        glorot_uniform = VarianceScaling(1.0, "fan_avg", "uniform")

        # Assemble hidden layers
        for _ in range(n_layers):
            layers.append(
                hk.Linear(
                    output_size=units,
                    w_init=glorot_uniform,
                    b_init=glorot_uniform
                )
            )
            layers.append(activation)

        # Add output layer without activation
        layers.extend([hk.Linear(
            output_size=1,
            w_init=glorot_uniform,
            b_init=glorot_uniform
        )
        ])

        self.mlp = hk.Sequential(layers)

    # Apply the model
    def __call__(self, x):
        return self.mlp(x)


class Physical_Base_LNN():
    def __init__(
        self,
        mnn_module,
        knn_module,
        dnn_module,
        mnn_settings,
        knn_settings,
        dnn_settings,
        mnn_optimizer,
        knn_optimizer,
        dnn_optimizer,
        info,
        activation,
        seed=0
    ):
        self.mnn_module = mnn_module
        self.knn_module = knn_module
        self.dnn_module = dnn_module
        self.mnn_settings = mnn_settings
        self.knn_settings = knn_settings
        self.dnn_settings = dnn_settings
        self.mnn_optimizer = mnn_optimizer
        self.knn_optimizer = knn_optimizer
        self.dnn_optimizer = dnn_optimizer
        self.info = info
        self.activation = activation
        self.key = jax.random.PRNGKey(seed)

    def _compile(self, settings, hk_module, rng, init_data, **kwargs):
        def forward_fn(x):
            """Forward pass"""
            module = hk_module(settings, self.activation, **kwargs)
            return module(x)

        net = hk.without_apply_rng(hk.transform(forward_fn))
        params = net.init(rng=rng, x=init_data)

        return params, net

    def gather(self):
        """Initialize M, K & D as separate networks"""
        mnn_init_data = jnp.zeros(
            self.mnn_settings["input_shape"], dtype=jnp.float32)
        mnn_params, mnn_net = self._compile(
            self.mnn_settings, self.mnn_module, rng=self.key, init_data=mnn_init_data)

        knn_init_data = jnp.zeros(
            self.knn_settings["input_shape"], dtype=jnp.float32)
        knn_params, knn_net = self._compile(
            self.knn_settings, self.knn_module, rng=self.key, init_data=knn_init_data)

        dnn_init_data = jnp.zeros(
            self.dnn_settings["input_shape"], dtype=jnp.float32)
        dnn_params, dnn_net = self._compile(
            self.dnn_settings, self.dnn_module, rng=self.key, init_data=dnn_init_data)

        # Assign attributes
        self.mnn_net = mnn_net
        self.knn_net = knn_net
        self.dnn_net = dnn_net
        self.mnn_init_params = mnn_params
        self.knn_init_params = knn_params
        self.dnn_init_params = dnn_params
        self.loss = self._loss()

        return self.mnn_net, self.knn_net, self.dnn_net

    def _update(self):
        """Update step"""
        loss_fn = self._loss()
        mnn_opt = self.mnn_optimizer
        knn_opt = self.knn_optimizer
        dnn_opt = self.dnn_optimizer
        mnn_opt_init_state = mnn_opt.init(self.mnn_init_params)
        knn_opt_init_state = knn_opt.init(self.knn_init_params)
        dnn_opt_init_state = dnn_opt.init(self.dnn_init_params)

        @jax.jit
        def update(mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, batch, count_idx):
            # Determine losses & gradients
            loss_value, (mnn_grads, knn_grads, dnn_grads) = jax.value_and_grad(
                loss_fn, [0, 1, 2])(mnn_params, knn_params, dnn_params, batch, count_idx)

            # Update network states
            mnn_updates, mnn_opt_state = mnn_opt.update(
                mnn_grads, mnn_opt_state, mnn_params)
            knn_updates, knn_opt_state = knn_opt.update(
                knn_grads, knn_opt_state, knn_params)
            dnn_updates, dnn_opt_state = dnn_opt.update(
                dnn_grads, dnn_opt_state, dnn_params)

            # Apply updates
            mnn_params = optax.apply_updates(mnn_params, mnn_updates)
            knn_params = optax.apply_updates(knn_params, knn_updates)
            dnn_params = optax.apply_updates(dnn_params, dnn_updates)

            return mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, loss_value

        self.update = update

        return mnn_opt_init_state, knn_opt_init_state, dnn_opt_init_state

    def init_step(self):
        """Initial step to get the initial optimizer state"""
        self.init_mnn_opt_state, self.init_knn_opt_state, self.init_dnn_opt_state = self._update()

        return self.mnn_init_params, self.knn_init_params, self.dnn_init_params, self.init_mnn_opt_state, self.init_knn_opt_state, self.init_dnn_opt_state

    def train_step(self, mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, batch, count_idx):
        """Define each training step"""
        mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, loss_value = self.update(
            mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, batch, count_idx)

        return mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, loss_value

    def test_step(self, mnn_params, knn_params, dnn_params, batch, count_idx):
        """Test step to gauge network"s accuracy"""
        batch_loss = self.loss(
            mnn_params, knn_params, dnn_params, batch, count_idx)
        return batch_loss

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

    def train(self, train_dataset, test_dataset, results=None, epochs=50, show_every=10):
        """train_dataset: training dataset (jnp.array),
           test_dataset: test_dataset (jnp.array),
           results: previous results if continuing training on a previously trained model (dict),
           epochs: number of desired training epochs (int),
           show_every: display training and test losses every n epochs (int)
        """
        # Initialize LNN & DNN
        _, _, _ = self.gather()

        # Iterator settings
        train_batch_size = self.mnn_settings["train_batch_size"]
        if self.mnn_settings["test_batch_size"] == -1:
            test_batch_size = test_dataset[0].shape[0]
        else:
            test_batch_size = self.mnn_settings["test_batch_size"]

        shuffle = self.mnn_settings["shuffle"]
        seed = self.mnn_settings["seed"]

        # Results data
        if results == None:
            results = {}

        _mnn_params, _knn_params, _dnn_params, _mnn_opt_state, _knn_opt_state, _dnn_opt_state = self.init_step()

        mnn_params = results.get("mnn_params", _mnn_params)
        knn_params = results.get("knn_params", _knn_params)
        dnn_params = results.get("dnn_params", _dnn_params)

        mnn_opt_state = results.get("mnn_opt_state", _mnn_opt_state)
        knn_opt_state = results.get("knn_opt_state", _knn_opt_state)
        dnn_opt_state = results.get("dnn_opt_state", _dnn_opt_state)

        best_mnn_params = results.get("best_mnn_params", None)
        best_knn_params = results.get("best_knn_params", None)
        best_dnn_params = results.get("best_dnn_params", None)

        _metrics = {"train_loss": [], "test_loss": []}
        metrics = results.get("metrics", _metrics)

        _best_loss = jnp.inf
        best_loss = results.get("best_loss", _best_loss)

        # Start training loop
        _start_epoch = 0
        start_epoch = results.get("last_epoch", _start_epoch)

        start_time = datetime.now()

        for step in range(start_epoch, start_epoch+epochs):
            # ------------------------train step
            train_epoch_loss = 0
            batches = 0
            train_batches = partial(
                self.DataIterator, batch=train_batch_size, shuffle=shuffle, seed=seed)

            for train_batch in train_batches(train_dataset):
                batches += 1
                mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, train_batch_loss = self.train_step(
                    mnn_params, knn_params, dnn_params, mnn_opt_state, knn_opt_state, dnn_opt_state, train_batch, step)
                train_epoch_loss += train_batch_loss

            train_epoch_loss /= (train_batch_size*batches)
            metrics["train_loss"].append(train_epoch_loss)

            # --------------------------test step
            test_epoch_loss = 0
            test_batches_counter = 0
            test_batches = partial(
                self.DataIterator, batch=test_batch_size, shuffle=shuffle, seed=seed)

            for test_batch in test_batches(test_dataset):
                test_batches_counter += 1
                test_batch_loss = self.test_step(
                    mnn_params, knn_params, dnn_params, test_batch, step)
                test_epoch_loss += test_batch_loss

            test_epoch_loss /= (test_batch_size*test_batches_counter)
            metrics["test_loss"].append(test_epoch_loss)

            # ----------------------------check loss
            if test_epoch_loss < best_loss:
                best_loss = test_epoch_loss
                best_mnn_params = jax.device_get(mnn_params)
                best_knn_params = jax.device_get(knn_params)
                best_dnn_params = jax.device_get(dnn_params)

            # --------------------------update the results dictionary
            if step % show_every == 0 or step == start_epoch:
                print(
                    f"Epoch: {step} | Train Loss: {train_epoch_loss:.8f} | Best Loss: {best_loss:.8f} | Test Loss: {test_epoch_loss:.8f}")
                print("---------------------------------")

        time_taken = datetime.now() - start_time

        self.best_mnn_params = jax.device_get(best_mnn_params)
        self.best_knn_params = jax.device_get(best_knn_params)
        self.best_dnn_params = jax.device_get(best_dnn_params)
        self.results = results

        results = {
            "metrics": jax.device_get(metrics),
            "best_loss": jax.device_get(best_loss),
            "mnn_params": jax.device_get(mnn_params),
            "knn_params": jax.device_get(knn_params),
            "dnn_params": jax.device_get(dnn_params),
            "best_mnn_params": jax.device_get(self.best_mnn_params),
            "best_knn_params": jax.device_get(self.best_knn_params),
            "best_dnn_params": jax.device_get(self.best_dnn_params),
            "mnn_opt_state": jax.device_get(mnn_opt_state),
            "knn_opt_state": jax.device_get(knn_opt_state),
            "dnn_opt_state": jax.device_get(dnn_opt_state),
            "last_epoch": jax.device_get(step+1),
            "time_taken": time_taken.total_seconds()/60,
            "mnn_settings": self.mnn_settings,
            "knn_settings": self.knn_settings,
            "dnn_settings": self.dnn_settings,
            "info": self.info
        }

        return results

    def plot_results(self, results):
        X = [step for step in range(results["last_epoch"])]
        Y1 = results["metrics"]["train_loss"]
        Y2 = results["metrics"]["test_loss"]
        plt.plot(X, Y1, label="Training Loss", color="blue")
        plt.plot(X, Y2, label="Test Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend()

    def save_model(self, results, model_name="", folder_name=""):
        DATA_DIR = os.path.join(folder_name, model_name)
        os.makedirs(DATA_DIR, exist_ok=True)
        model_fn = DATA_DIR + "/" + "model.pkl"
        metrics_fn = DATA_DIR + "/" + "metrics.pkl"

        pickle.dump(results, open(model_fn, "wb"))
        pickle.dump(results["metrics"], open(metrics_fn, "wb"))

        return model_fn, metrics_fn

    @staticmethod
    def load_model(file_name):

        with open(file_name, "rb") as file_cont:
            data = pickle.load(file_cont)

        return data


class Physical_Damped_LNN(Physical_Base_LNN):
    def __init__(
        self,
        mnn_module,
        knn_module,
        dnn_module,
        mnn_settings,
        knn_settings,
        dnn_settings,
        mnn_optimizer,
        knn_optimizer,
        dnn_optimizer,
        info,
        activation,
        seed=0
    ):
        super().__init__(
            mnn_module,
            knn_module,
            dnn_module,
            mnn_settings,
            knn_settings,
            dnn_settings,
            mnn_optimizer,
            knn_optimizer,
            dnn_optimizer,
            info,
            activation,
            seed)

    def _eom(self):
        def eom(M, K, D, L, x, f):
            q, q_t = jnp.split(x, 2, axis=-1)

            MQ = jax.hessian(M, 1)(q, q_t)  # m
            KQ = jax.jacobian(K, 0)(q, q_t)  # k
            CQ = jax.jacobian(D, 0)(q_t)  # c
            # ddL/dqdq_t term in Lagrangian formulation for eom
            GC1 = jax.jacfwd(jax.jacrev(L, 1), 0)(q, q_t)
            GCQ = jnp.tensordot(GC1, q_t, axes=1)

            S = KQ - CQ - GCQ - f
            invM = jnp.linalg.pinv(MQ)
            q_tt = jnp.tensordot(S, invM, axes=1)

            result = jnp.concatenate([q_t, q_tt], axis=-1)

            return q_tt

        return eom

    def _dynamics(self):
        mnn_net = self.mnn_net
        knn_net = self.knn_net
        dnn_net = self.dnn_net

        qmax = jnp.array([self.info["qmax"]], dtype=jnp.float32)
        qmin = jnp.array([self.info["qmin"]], dtype=jnp.float32)
        q_dmax = jnp.array([self.info["qdmax"]], dtype=jnp.float32)
        q_dmin = jnp.array([self.info["qdmin"]], dtype=jnp.float32)

        def dynamics(mnn_params, knn_params, dnn_params):
            def zeros(x):
                """Enforce boundary conditions"""
                q, q_t = jnp.split(x, 2, -1)
                z = jnp.zeros_like(x)
                z_0 = jnp.zeros_like(q)

                m = mnn_net.apply(mnn_params, z)
                k = knn_net.apply(knn_params, z)
                d = dnn_net.apply(dnn_params, z_0)

                dm = jax.jacobian(mnn_net.apply, 1)(mnn_params, z)
                dk = jax.jacobian(knn_net.apply, 1)(knn_params, z)
                dd = jax.jacobian(dnn_net.apply, 1)(dnn_params, z_0)

                ll = jnp.concatenate([m, k, d], axis=-1)
                ddl = jnp.concatenate([dm, dk, dd], axis=-1)

                return ll, ddl

            def lagrangian(qq, qq_t):
                """LNN forward step with normalized inputs"""
                assert qq.shape == (1,)

                qn = 2*(qq - qmin)/(qmax - qmin) - 1
                q_tn = 2*(qq_t - q_dmin)/(q_dmax - q_dmin) - 1

                state = jnp.concatenate([qn, q_tn], axis=-1)
                M = mnn_net.apply(mnn_params, state)
                K = knn_net.apply(knn_params, state)

                L = M - K

                return jnp.squeeze(L)

            def mass(qq, qq_t):
                """MNN forward step with normalized inputs"""
                assert qq.shape == (1,)

                qn = 2*(qq - qmin)/(qmax - qmin) - 1
                q_tn = 2*(qq_t - q_dmin)/(q_dmax - q_dmin) - 1

                state = jnp.concatenate([qn, q_tn], axis=-1)
                M = mnn_net.apply(mnn_params, state)

                return jnp.squeeze(M)

            def stiffness(qq, qq_t):
                """KNN forward step with normalized inputs"""
                assert qq.shape == (1,)

                qn = 2*(qq - qmin)/(qmax - qmin) - 1
                q_tn = 2*(qq_t - q_dmin)/(q_dmax - q_dmin) - 1

                state = jnp.concatenate([qn, q_tn], axis=-1)
                K = knn_net.apply(knn_params, state)

                return jnp.squeeze(K)

            def dissipation(qq_t):
                """DNN forward step with normalized inputs"""
                assert qq_t.shape == (1,)

                q_tn = 2*(qq_t - q_dmin)/(q_dmax - q_dmin) - 1
                D = dnn_net.apply(dnn_params, q_tn)

                return jnp.squeeze(D)

            def damp_sym(x):
                """Enforce symmetry about qdot"""
                qq_t = jnp.split(x, 2, -1)[1]
                q_tn = 2*(qq_t - q_dmin)/(q_dmax - q_dmin) - 1

                D_sym = jax.jacrev(jax.hessian(dnn_net.apply, 1), 1)(
                    dnn_params, q_tn)

                return jnp.squeeze(D_sym)

            return mass, stiffness, dissipation, lagrangian, zeros, damp_sym

        return dynamics

    def _loss(self):
        eom = self._eom()
        dynamics = self._dynamics()

        @jax.jit
        def loss_fn(mnn_params,
                    knn_params,
                    dnn_params,
                    batch,
                    count_idx):
            mass, stiffness, dissipation, lagrangian, zeros, damp_sym = dynamics(
                mnn_params, knn_params, dnn_params)
            q, f, q_d = batch

            # Calculate loss terms
            pred = jax.vmap(
                partial(eom, mass, stiffness, dissipation, lagrangian))(q, f)
            ll, ddl = jax.vmap(partial(zeros))(q)
            D_sym = jax.vmap(partial(damp_sym))(q)

            # MSE Calculation
            # Acceleration
            pred_mse = jnp.mean(jnp.square(
                pred.flatten() - q_d[:, -1].flatten()))
            # Known Prior: L(0, 0) = 0 & D(0) = 0
            ll_mse = jnp.mean(jnp.square(ll.flatten()))
            ddl_mse = jnp.mean(jnp.square(ddl.flatten()))
            # Constraint: Symmetric D()
            D_sym_mse = jnp.mean(jnp.square(D_sym.flatten()))

            # Losses
            loss = pred_mse + ll_mse + ddl_mse + 1.0*D_sym_mse

            return loss

        self.loss = loss_fn

        return self.loss

    def _predict(self, results):
        mnn_params = results["best_mnn_params"]
        knn_params = results["best_knn_params"]
        dnn_params = results["best_dnn_params"]
        info = results["info"]

        qmax = info["qmax"]
        qmin = info["qmin"]
        q_dmax = info["qdmax"]
        q_dmin = info["qdmin"]

        mnn_net, knn_net, dnn_net = self.gather()
        eom = self._eom()
        dynamics = self._dynamics()
        mass, stiffness, dissipation, lagrangian, _, _ = dynamics(
            mnn_params, knn_params, dnn_params)

        @jax.jit
        def predict_accel(q, f):
            pred = jax.vmap(
                partial(eom, mass, stiffness, dissipation, lagrangian))(q, f)
            return pred

        @jax.jit
        def predict_energy(q, q_t):
            qn = 2*(q - qmin)/(qmax - qmin) - 1
            q_tn = 2*(q_t - q_dmin)/(q_dmax - q_dmin) - 1

            state = jnp.concatenate([qn, q_tn], axis=-1)

            M = self.mnn_net.apply(mnn_params, state)
            K = self.knn_net.apply(knn_params, state)
            D = self.dnn_net.apply(dnn_params, q_tn)
            result = M, K, D

            return result

        return predict_accel, predict_energy

    def evaluate(self, results, n_points=40, file_name="MKD"):
        _, pred_energy = self._predict(results)
        limq1, limq2, limqd1, limqd2 = self.info["qmin"], self.info["qmax"], self.info["qdmax"], self.info["qdmin"]

        qa, qda = jnp.linspace(limq1, limq2, n_points), jnp.linspace(
            limqd1, limqd2, n_points)
        qaa, qdaa = jnp.meshgrid(qa, qda)

        # Get all energy functions here
        M, K, D = pred_energy(qaa.reshape(-1, 1, 1), qdaa.reshape(-1, 1, 1))
        L = M - K

        fig = plt.figure(figsize=(12, 12), tight_layout=True)
        fig.suptitle(f"Final Test Loss: {results['best_loss']:.3e}")

        # --------------------------------- FUNCTIONS -------------------------------- #

        # -------------------------------- Lagrangian
        ax = fig.add_subplot(221, projection="3d")
        m = ax.plot_surface(qaa, qdaa, L.reshape(qaa.shape), cmap="RdGy", lw=0)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\dot{q}$")
        ax.set_zlabel(r"$\mathcal{L}_{NN}$", fontsize=16, labelpad=3)
        ax.set_title(f"Lagrangian")
        fig.colorbar(m, ax=ax, shrink=0.3, pad=0.1)

        # ---------------------------------- Mass
        ax = fig.add_subplot(222, projection="3d")
        m = ax.plot_surface(qaa, qdaa, M.reshape(qaa.shape), cmap="PiYG", lw=0)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\dot{q}$")
        ax.set_zlabel(r"$\mathcal{M}_{NN}$", fontsize=16, labelpad=3)
        ax.set_title("Mass")
        fig.colorbar(m, ax=ax, shrink=0.3, pad=0.1)

        # ---------------------------------- Stiffness
        ax = fig.add_subplot(223, projection="3d")
        m = ax.plot_surface(qaa, qdaa, K.reshape(qaa.shape), cmap="PiYG", lw=0)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\dot{q}$")
        ax.set_zlabel(r"$\mathcal{K}_{NN}$", fontsize=16, labelpad=3)
        ax.set_title("Stiffness")
        fig.colorbar(m, ax=ax, shrink=0.3, pad=0.1)

        # ---------------------------------- Damping
        ax = fig.add_subplot(224, projection="3d")
        m = ax.plot_surface(qaa, qdaa, D.reshape(qaa.shape), cmap="PiYG", lw=0)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\dot{q}$")
        ax.set_zlabel(r"$\mathcal{D}_{NN}$", fontsize=16, labelpad=3)
        ax.set_title("Damping")
        fig.colorbar(m, ax=ax, shrink=0.3, pad=0.1)
        # fig.savefig(f"./Physical_LNN/{file_name}-LD.png")

        # --------------------------------- ENERGIES --------------------------------- #
        fig = plt.figure(figsize=(10, 10), tight_layout=True)

        # --------------------------------- Stiffness
        K = jax.vmap(jax.jacrev(pred_energy, 0))(
            qa.reshape(-1, 1, 1), qda.reshape(-1, 1, 1))[1]
        ax = fig.add_subplot(221)
        ax.plot(qa, -K.squeeze(), lw=0.8)
        ax.set_title("Predicted Spring Force")
        ax.set_xlabel(r"$q \ (m)$", fontsize=12)
        ax.set_ylabel(
            r"$-\frac{\partial \mathcal{L}_{NN}}{\partial q}$", fontsize=18, labelpad=7)

        # ---------------------------------- Damping
        C = jax.vmap(jax.jacrev(pred_energy, 1))(
            qa.reshape(-1, 1, 1), qda.reshape(-1, 1, 1))[2]
        ax = fig.add_subplot(222)
        ax.plot(qda, C.squeeze(), lw=0.8)
        ax.set_title("Predicted Damping Force")
        ax.set_ylabel(r"$\dot{q} \ (m \ s^{-1})$ ", fontsize=12)
        ax.set_ylabel(
            r"$\frac{\partial \mathcal{D}_{NN}}{\partial \dot{q}}$", fontsize=18, labelpad=7)

        # ----------------------------------- Mass
        M = jax.vmap(jax.hessian(pred_energy, 1))(
            qa.reshape(-1, 1, 1), qda.reshape(-1, 1, 1))[0]
        ax = fig.add_subplot(223)
        m = ax.plot(qa, M.reshape(qda.shape), lw=0.8)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(
            r"$\frac{\partial}{\partial t}(\frac{\partial \mathcal{L}_{NN}}{\partial q})$", fontsize=16, labelpad=3)
        ax.set_title(f"Mass")

        ax = fig.add_subplot(224)
        m = ax.plot(qda, M.reshape(qda.shape), lw=0.8)
        ax.set_xlabel(r"$\dot{q}$")
        ax.set_ylabel(
            r"$\frac{\partial}{\partial t}(\frac{\partial \mathcal{L}_{NN}}{\partial q})$", fontsize=16, labelpad=3)
        ax.set_title(f"Mass")

        # fig.savefig(f"./Physical_LNN/{file_name}-MKD.png")

    def evaluate_S_curves(self, results, dataset, file_name="S-curves"):
        pred_accel, _ = self._predict(results)
        Q, F, DQ, W = dataset

        # Compute accelerations from LNN
        pred = pred_accel(Q, F)

        # Reshape for computations
        DQtt_out, pred_out, F_out = [], [], []
        n_seqs = W.shape[0]
        n_t_samples = self.info["n_t_samples"]
        for i in range(n_seqs):
            DQtt_out.append(DQ[i*n_t_samples:(i+1)*n_t_samples])
            F_out.append(F[i*n_t_samples:(i+1)*n_t_samples])
            pred_out.append(pred[i*n_t_samples:(i+1)*n_t_samples])
        DQtt = jnp.array(DQtt_out)
        F_out = jnp.array(F_out)
        pred = jnp.array(pred_out)
        W_rnd = jnp.around(W, decimals=1).reshape(-1, 1)

        # Store maximums
        F_max = np.max(F_out, axis=1)[:].reshape(-1, 1)
        pred_qtt_max = np.max(pred, axis=1)[:, 1].reshape(-1, 1)
        DQtt_max = np.max(DQtt, axis=1)[:, 1].reshape(-1, 1)

        # Combine data for plotting
        data = np.concatenate((W_rnd, F_max, pred_qtt_max, DQtt_max), axis=-1)
        data = data[data[:, -1].argsort()]
        W_rnd, F_max, pred_qtt_max, DQtt_max = data[:,
                                                    0], data[:, 1], data[:, 2], data[:, 3]

        # Assemble data per forcing frequency
        data_dict = {}
        for i in range(W_rnd.shape[0]):
            data_dict.setdefault(f"{W_rnd[i]}", [])
            data_dict[f"{W_rnd[i]}"].append(F_max[i])
            data_dict[f"{W_rnd[i]}"].append(pred_qtt_max[i])
            data_dict[f"{W_rnd[i]}"].append(DQtt_max[i])

        # Create figures
        ncols = 3
        nrows = int(np.ceil(len(data_dict)/ncols))
        n_fig = 1

        fig = plt.figure(figsize=(4*ncols, 4*nrows), tight_layout=True)
        for w, v in sorted(data_dict.items()):
            data_dict[w] = jnp.array(v)

            axs = plt.subplot(nrows, ncols, n_fig)
            n_fig = n_fig+1

            axs.scatter(v[0::3], v[1::3], marker=".", lw=0.8, label="LNN")
            axs.scatter(v[0::3], v[2::3], marker="x",
                        lw=0.8, label="Ground-truth")
            axs.set_title(f"S-curves at $\omega$={float(w):.2f}")
            axs.set_xlabel(r"Base acceleration $\ddot{y} \ (m/s^2)$")
            axs.set_ylabel(r"Response Acceleration $\ddot{X} \ (m/s^2)$")
            axs.legend()

        # fig.savefig(f"./Physical_LNN/{file_name}-S_curve.png")
