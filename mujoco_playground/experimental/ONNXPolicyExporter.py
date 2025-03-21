# onnx_export_class.py

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import pickle

import jax
import jax.numpy as jp
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as rt

import matplotlib.pyplot as plt

# Brax / custom imports
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params
from mujoco_playground import locomotion


class ONNXPolicyExporter:
    """
    This class encapsulates:
      1) Loading a trained Brax PPO checkpoint (params.pkl) [optional if you already have an inference_fn]
      2) Building a TensorFlow model (Keras) that replicates the MLP
      3) Transferring JAX weights -> TensorFlow
      4) Exporting the TF model to ONNX
      5) (Optional) Running inference in JAX vs. TF vs. ONNX to compare

    You can also directly set self.inference_fn if you already have a JAX inference function.
    """

    def __init__(self, env_name, checkpoint_path=None, output_path="policy.onnx"):
        """
        Args:
            env_name (str): Name of the Brax environment.
            checkpoint_path (str, optional): Path to the 'params.pkl' checkpoint.
                If None, you can later call set_inference_fn() with your own function.
            output_path (str): Where to save the ONNX file.
        """
        self.env_name = env_name
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path

        # 1) Load environment for size info
        self.env_cfg = locomotion.get_default_config(env_name)
        self.env = locomotion.load(env_name, config=self.env_cfg)
        self.obs_size = self.env.observation_size
        self.act_size = self.env.action_size

        # 2) Create the network factory for PPO (if needed)
        #    We'll store it here if we want to load checkpoint and auto-build the network
        self.ppo_params = locomotion_params.brax_ppo_config(env_name)
        self.network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **self.ppo_params.network_factory,
            preprocess_observations_fn=running_statistics.normalize,  # for normalizing obs
        )
        self.ppo_network = self.network_factory(self.obs_size, self.act_size)

        # Will hold data once loaded
        self.params = None           # The (normalizer_params, policy_params) from checkpoint
        self.inference_fn = None     # JAX inference function
        self.tf_policy_network = None  # TF Keras model

    def set_inference_fn(self, inference_fn):
        """
        Provide a custom inference function if you already have it externally.

        Args:
            inference_fn: A callable of form f(obs_dict, rng) -> (actions, extra), typically JAX-jitted.
        """
        self.inference_fn = inference_fn
        print("Custom JAX inference function assigned to self.inference_fn.")

    def load_checkpoint(self):
        """
        Load the params.pkl checkpoint and build the JAX inference function (if checkpoint_path is provided).
        """
        if self.checkpoint_path:
            with open(self.checkpoint_path, 'rb') as f:
                ckpt_data = pickle.load(f)
            print("Checkpoint keys:", ckpt_data.keys())

            # JAX PPO expects (normalizer_params, policy_params)
            self.params = (ckpt_data["normalizer_params"], ckpt_data["policy_params"])

        if not self.params:
            raise ValueError("No params loaded. Either provide a checkpoint_path or set self.params manually")
        
        # Build the inference function
        make_inference_fn = ppo_networks.make_inference_fn(self.ppo_network)
        self.inference_fn = make_inference_fn(self.params, deterministic=True)
        print("Checkpoint loaded and JAX inference_fn created from checkpoint.")

    def build_tf_model(self):
        """
        Build an equivalent TF Keras MLP model.
        This includes observation normalization if needed.
        """
        if self.params is None:
            # It's possible that you only want to set_inference_fn without a checkpoint.
            # But for the TF model, we specifically need normalizer_params from self.params
            raise ValueError(
                "Must either load_checkpoint() or otherwise set self.params. "
                "The TF model requires normalizer_params to handle obs normalization."
            )

        # Normalizer stats
        normalizer = self.params[0]
        mean = normalizer.mean["state"]
        std = normalizer.std["state"]

        # Convert them to TF variables
        mean_std = (
            tf.convert_to_tensor(mean, dtype=tf.float32),
            tf.convert_to_tensor(std, dtype=tf.float32),
        )

        # The hidden layer sizes from PPO config
        hidden_layer_sizes = self.ppo_params.network_factory.policy_hidden_layer_sizes

        class MLP(tf.keras.Model):
            def __init__(
                self,
                layer_sizes,
                activation=tf.nn.relu,
                kernel_init="lecun_uniform",
                activate_final=False,
                bias=True,
                layer_norm=False,
                mean_std=None,
            ):
                super().__init__()
                self.mean = None
                self.std = None
                if mean_std is not None:
                    self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                    self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)

                self.mlp_block = tf.keras.Sequential(name="MLP_0")
                for i, size in enumerate(layer_sizes):
                    dense_layer = tf.keras.layers.Dense(
                        size,
                        activation=activation,
                        kernel_initializer=kernel_init,
                        name=f"hidden_{i}",
                        use_bias=bias,
                    )
                    self.mlp_block.add(dense_layer)
                    if layer_norm:
                        self.mlp_block.add(tf.keras.layers.LayerNormalization(name=f"layer_norm_{i}"))

                if not activate_final and self.mlp_block.layers:
                    # Remove activation from the last layer if set
                    if hasattr(self.mlp_block.layers[-1], "activation"):
                        self.mlp_block.layers[-1].activation = None

            def call(self, inputs):
                # Normalization
                if self.mean is not None and self.std is not None:
                    inputs = (inputs - self.mean) / self.std
                logits = self.mlp_block(inputs)
                # PPO splits final layer into (loc, scale). We only use loc -> tanh
                loc, _ = tf.split(logits, 2, axis=-1)
                return tf.tanh(loc)

        # Construct
        self.tf_policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [self.act_size * 2],
            activation=tf.nn.swish,
            kernel_init="lecun_uniform",
            layer_norm=False,
            mean_std=mean_std,
        )

        # Build once with a dummy input
        example_input = tf.zeros((1, self.obs_size["state"][0]))
        _ = self.tf_policy_network(example_input)
        print("TF Keras model built.")

    def transfer_weights_jax_to_tf(self):
        """
        Transfers the JAX policy params into the built TF model's layers.
        """
        if self.tf_policy_network is None:
            raise ValueError("Must call build_tf_model() first to create TF model.")

        if self.params is None:
            raise ValueError("No JAX params set. Either load_checkpoint() or set self.params manually.")

        # Our JAX policy params are at: self.params[1]['params'] => the MLP dictionary
        jax_params = self.params[1]["params"]
        tf_model = self.tf_policy_network

        def transfer_weights(params_dict, keras_model):
            for layer_name, layer_dict in params_dict.items():
                try:
                    tf_layer = keras_model.get_layer("MLP_0").get_layer(name=layer_name)
                except ValueError:
                    print(f"[WARN] Layer {layer_name} not found in Keras model.")
                    continue
                if isinstance(tf_layer, tf.keras.layers.Dense):
                    kernel = np.array(layer_dict["kernel"])
                    bias = np.array(layer_dict["bias"])
                    print(f"Transferring to layer {layer_name}: kernel {kernel.shape}, bias {bias.shape}")
                    tf_layer.set_weights([kernel, bias])
                else:
                    print(f"[WARN] Unhandled layer type for {layer_name}: {type(tf_layer)}")

        transfer_weights(jax_params, tf_model)
        print("JAX weights transferred to the TF model successfully.")

    def export_to_onnx(self):
        """
        Exports the TF model to ONNX using tf2onnx.
        """
        if self.tf_policy_network is None:
            raise ValueError("TF policy network not built. Call build_tf_model() first.")

        # Prepare input signature
        spec = [tf.TensorSpec(shape=(1, self.obs_size["state"][0]), dtype=tf.float32, name="obs")]
        self.tf_policy_network.output_names = ["continuous_actions"]

        # Convert to ONNX
        print(f"Exporting to ONNX: {self.output_path}")
        _model_proto, _ = tf2onnx.convert.from_keras(
            self.tf_policy_network,
            input_signature=spec,
            opset=11,
            output_path=self.output_path,
        )
        print(f"ONNX file saved as {self.output_path}")

    def compare_inference(self):
        """
        Runs a quick check comparing JAX vs. TF vs. ONNX outputs on dummy input.
        """
        if not self.inference_fn:
            raise ValueError(
                "No JAX inference_fn is set. Either load_checkpoint() or set_inference_fn() first."
            )
        if self.tf_policy_network is None:
            raise ValueError("TF policy network not built. Call build_tf_model() first.")

        # 1) TF inference
        test_input_tf = tf.ones((1, self.obs_size["state"][0]), dtype=tf.float32)
        tf_output = self.tf_policy_network(test_input_tf).numpy()[0]

        # 2) ONNX inference
        providers = ["CPUExecutionProvider"]
        sess = rt.InferenceSession(self.output_path, providers=providers)
        onnx_input = {"obs": np.ones((1, self.obs_size["state"][0]), dtype=np.float32)}
        onnx_output = sess.run(["continuous_actions"], onnx_input)[0][0]

        # 3) JAX inference
        test_input_jax = {
            "state": jp.ones(self.obs_size["state"]),
            "privileged_state": jp.zeros(self.obs_size["privileged_state"]),
        }
        jax_output, _ = self.inference_fn(test_input_jax, jax.random.PRNGKey(0))

        print("\n--- Comparison of outputs on dummy input ---")
        print(f"TF output:   {tf_output}")
        print(f"ONNX output: {onnx_output}")
        print(f"JAX output:  {jax_output}")

        # Simple plot
        plt.plot(tf_output, label="TF")
        plt.plot(onnx_output, label="ONNX")
        plt.plot(jax_output, label="JAX")
        plt.legend()
        plt.title("Comparison of policy outputs")
        plt.show()


# Example usage (if run directly)
if __name__ == "__main__":
    env_name = "BerkeleyHumanoidJoystickFlatTerrain"
    ckpt_path = "/path/to/BerkeleyHumanoidJoystickFlatTerrain-20250111-001442/params.pkl"
    output_path = "bh_policy.onnx"

    # Option A: Use the checkpoint
    exporter = ONNXPolicyExporter(env_name, checkpoint_path=ckpt_path, output_path=output_path)
    exporter.load_checkpoint()            # loads self.params and self.inference_fn
    exporter.build_tf_model()            # builds the Keras model
    exporter.transfer_weights_jax_to_tf() # copies JAX -> Keras
    exporter.export_to_onnx()
    exporter.compare_inference()

    # Option B: If you already have an inference_fn and params, do something like:
    # new_exporter = ONNXPolicyExporter(env_name)
    # new_exporter.params = your_params                  # So you can still build TF model if it needs mean/std
    # new_exporter.set_inference_fn(your_inference_fn)   # Provide your external JAX inference function
    # new_exporter.load_checkpoint()  # Or, Create inference_fn from params
    # new_exporter.build_tf_model()
    # new_exporter.transfer_weights_jax_to_tf()
    # new_exporter.export_to_onnx()
    # new_exporter.compare_inference()
