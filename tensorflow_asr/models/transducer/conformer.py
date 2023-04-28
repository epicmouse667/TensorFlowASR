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
import collections
import tensorflow as tf
import numpy as np

from tensorflow_asr.models.encoders.conformer import L2, ConformerEncoder,ConformerDecoder
from tensorflow_asr.models.transducer.base_transducer import Transducer
from tensorflow_asr.utils import data_util, layer_util, math_util, shape_util


Hypothesis = collections.namedtuple("Hypothesis", ("prediction"))

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
        encoded = tf.reshape(encoded, [1, 1, -1]) 
        decoded = self.decoder(encoded)
        decoded = tf.reshape(decoded, shape=[-1])
        return decoded

     @tf.function
    def recognize(
        self,
        inputs: Dict[str, tf.Tensor],
    ):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of extracted features
            input_length (tf.Tensor): a batch of extracted features length

        Returns:
            tf.Tensor: a batch of decoded transcripts

        Note that: 
        We will have padding on both sides of the input vector. 
        The padding will be the corresponding log_mel vector of silence.
        """
        config = Config(config)
        head_redundancy = config.infer_config.head_redundancy
        tail_redundancy = config.infer_config.tail_redundancy
        effective_len = config.infer_config.effective_len
        silence_audio_path = config.infer_config.silence_audio_path

        silence = tf.conver_to_tensor(np.load(silence_audio_path))
        silence = self.encoder(silence,traing=Flase)
        encoded = self.encoder(inputs["inputs"], training=False)
        encoded_length = math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        return self._perform_streaming_in_batch(
            encoded=encoded, encoded_length=encoded_length,
            head=head_redundancy,tail=tail_redundancy,
            silence=silence,effective_len=effective_len)


    def _perform_greedy(
            self,
            encoded: tf.Tensor,
            encoded_length: tf.Tensor,
            parallel_iterations: int = 10,
            swap_memory: bool = False,
            tflite: bool = False,
        ):
            with tf.name_scope(f"{self.name}_greedy"):
                time = tf.constant(0, dtype=tf.int32)
                total = encoded_length

                hypothesis = Hypothesis(
                    prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=total,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                    ),
                )
                def condition(_time,_):
                    return tf.less(_time, total)

                def body(_time,_hypothesis):
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
                    _hypothesis = Hypothesis(prediction=_prediction)

                    return _time + 1,_hypothesis

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
  
    def _streaming_inference(
        self,
        encoded,
        encoded_length,
        head,
        tail,
        silence,
        effective_len,
        parallel_iterations:int = 10,
        swap_memory: bool = False,
        tflite:bool = False
    ):
    """
    Note:
    This function will do straming inference. The function will have a sliding window
    (win_len = head_redundancy+effective_len+tail_redundancy), and every time the window move forward, it will take a segment of
     length [effective_len] frames and feed it to _perform_greedy() which will output the decoded transcript. Then the window 
     will hop forward [hop_len] frames. Once the window finished sliding, this function will collect all the 
     transciprts(with length of [effective_len]) and concatenate all of thems to form a complete transcript.
    """
        with tf.name_scope(f"{self.name}_streaming_infer"):
            hypothesis = Hypothesis(
                    prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([effective_len]),
                    ),
                )

                return Hypothesis(
                    prediction=hypothesis.prediction.stack(),
                )
            win_len = head+effective_len+tail 
            '''
             window made up by three parts, which is head,effective_len,tail.
             we feed the whole window to perform_greedy,but we only care about the effective_len in the middle.
            '''
            encoded = tf.concat([silence[:head+1],encoded,silence[:tail+1]])

            time = tf.constant(0, dtype=tf.int32)
            def condition(_time):
                tf.less(_time+effective_len+tail,encoded_length+win_len)
            def body(_time,_hypothesis):
                _infer = _perform_greedy(
                    encoded=encoded[_time,_time+win_len],
                    enccoded_length=tf.constant(win_len,dtype=tf.int32),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                    tflite=tflite
                )
                if tail==0:
                    _prediction = _hypothesis.prediction.write(_time, _infer[head:])
                else:
                    _prediction = _hypothesis.prediction.write(_time, _infer[head,-tail])
                _hypothesis = Hypothesis(prediction=_prediction)
                return _time+effective_len,_hypothesis
            time, hypothesis = tf.while_loop(
                    condition,
                    body,
                    loop_vars=[time, hypothesis],
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
            return Hypothesis(
                    prediction=hypothesis.prediction.concat(),
                )

    def _perform_straming_inference_in_batch(
        self,
        encoded,
        encoded_length,
        head,
        tail,
        silence,
        parallel_iterations:int = 10,
        swap_memory: bool = Flase,
        tflite:bool = False
        ):
         with tf.name_scope(f"{self.name}_perform_streaming_inference_in_batch"):
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
                    hypothesis = self._streaming_inference(
                        encoded=encoded[batch],
                        encoded_length=encoded_length[batch],
                        head=head,
                        tail=tail,
                        silence=silence,
                        effective_len=effective_len,
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
