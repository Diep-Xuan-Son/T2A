import jax
import jax.numpy as jnp
import haiku as hk
from hifigan.model import Generator
import pickle

@hk.transform_with_state
def forward_hifi(x):
	# with open(f"{DIR}/weights/hifigan/config.json") as f:
	# 	data = f.read()
	# json_config = json.loads(data)
	# h = AttrDict(json_config)
	net = Generator()
	return net(x)

forward_hifi = hk.transform(forward_hifi)
rng = next(hk.PRNGSequence(42))
mel = jnp.ones([1, 80, 80])
with open(f"./weights/hifigan/hk_hifi.pickle", "rb") as f:
	params_hifi = pickle.load(f)

import functools
inference = functools.partial(forward_hifi.apply, params_hifi, None)

# jax2tf enables us to easily go from JAX -> TensorFlow.
import tensorflow as tf
from jax.experimental import jax2tf

inference_tf = jax2tf.convert(inference, enable_xla=False)
inference_tf = tf.function(inference_tf, autograph=False)

# tf2onnx allows TF programs to be staged out as onnx protos.
import tf2onnx

inference_onnx = tf2onnx.convert.from_function(inference_tf, input_signature=[tf.TensorSpec([1, 1])])
model_proto, external_tensor_storage = inference_onnx
print(external_tensor_storage)