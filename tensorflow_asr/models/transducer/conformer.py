# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow_asr.models.encoders.conformer import L2, ConformerEncoder,ConformerDecoder
from tensorflow_asr.models.transducer.base_transducer import Transducer
import tensorflow as tf


class Conformer(Transducer):
    def __init__(
        self,
        vocabulary_size: int,
        encoder_subsampling: dict,
        encoder_positional_encoding: str = "sinusoid",
        encoder_dmodel: int = 144,
        encoder_num_blocks: int = 16,
        encoder_head_size: int = 36,
        encoder_num_heads: int = 4,
        encoder_mha_type: str = "relmha",
        encoder_kernel_size: int = 32,
        encoder_depth_multiplier: int = 1,
        encoder_fc_factor: float = 0.5,
        encoder_dropout: float = 0,
        encoder_trainable: bool = True,
        prediction_embed_dim: int = 512,
        prediction_embed_dropout: int = 0,
        prediction_num_rnns: int = 1,
        prediction_rnn_units: int = 320,
        prediction_rnn_type: str = "lstm",
        prediction_rnn_implementation: int = 2,
        prediction_layer_norm: bool = True,
        prediction_projection_units: int = 0,
        prediction_trainable: bool = True,
        joint_dim: int = 1024,
        joint_activation: str = "tanh",
        prejoint_linear: bool = True,
        postjoint_linear: bool = False,
        joint_mode: str = "add",
        joint_trainable: bool = True,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name: str = "conformer",
        **kwargs,
    ):
        super(Conformer, self).__init__(
            encoder=ConformerEncoder(
                subsampling=encoder_subsampling,
                positional_encoding=encoder_positional_encoding,
                dmodel=encoder_dmodel,
                num_blocks=encoder_num_blocks,
                head_size=encoder_head_size,
                num_heads=encoder_num_heads,
                mha_type=encoder_mha_type,
                kernel_size=encoder_kernel_size,
                depth_multiplier=encoder_depth_multiplier,
                fc_factor=encoder_fc_factor,
                dropout=encoder_dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                trainable=encoder_trainable,
                name=f"{name}_encoder",
            ),
            decoder=ConformerDecoder(
                vocabulary_size=vocabulary_size,
                filters=1,
                kernel_regularizer=kernel_regularizer,
                kernel_size=encoder_kernel_size,
                bias_regularizer=bias_regularizer,
                name=f"{name}_decoder",
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=prediction_embed_dim,
            embed_dropout=prediction_embed_dropout,
            num_rnns=prediction_num_rnns,
            rnn_units=prediction_rnn_units,
            rnn_type=prediction_rnn_type,
            rnn_implementation=prediction_rnn_implementation,
            layer_norm=prediction_layer_norm,
            projection_units=prediction_projection_units,
            prediction_trainable=prediction_trainable,
            joint_dim=joint_dim,
            joint_activation=joint_activation,
            prejoint_linear=prejoint_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            joint_trainable=joint_trainable,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
            **kwargs,
        )
        self.dmodel = encoder_dmodel
        # self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor 
        self.time_reduction_factor = 1
        
def decoder_inference(
        self,
        encoded: tf.Tensor,
        tflite: bool = False,
    ):
    decoded = self.decoder(encoded)
    return decoded

def _perform_greedy(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
        tflite: bool = False,
    ):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length
            def condition(_time):
                return tf.less(_time, total)

            def body(_time):
                ytu = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    tflite=tflite,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                # something is wrong with tflite that drop support for tf.cond
                # def equal_blank_fn(): return _hypothesis.index, _hypothesis.states
                # def non_equal_blank_fn(): return _predict, _states  # update if the new prediction is a non-blank
                # _index, _states = tf.cond(tf.equal(_predict, blank), equal_blank_fn, non_equal_blank_fn)

                _prediction = _hypothesis.prediction.write(_time, _predict)

                return _time + 1

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                prediction=hypothesis.prediction.stack(),
            )

def _perform_greedy_batch(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([None]),
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = self._perform_greedy(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = math_util.pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())
