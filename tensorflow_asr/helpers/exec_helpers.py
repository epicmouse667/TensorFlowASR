from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset

from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import file_util, app_util
from tensorflow_asr.configs.config import Config
exec_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(exec_dir, '..', '..', 'examples', 'conformer')
DEFAULT_YAML = os.path.join(config_dir, 'config.yml')

logger = tf.get_logger()

def run_testing(
    model: BaseModel,
    test_dataset: ASRSliceDataset,
    test_data_loader: tf.data.Dataset,
    output: str,
    config:str=DEFAULT_YAML
):
    with file_util.save_file(file_util.preprocess_paths(output)) as filepath:
        overwrite = True
        if tf.io.gfile.exists(filepath):
            overwrite = input(f"Overwrite existing result file {filepath} ? (y/n): ").lower() == "y"
        # if overwrite:
        #     results = model.predict(test_data_loader, verbose=1)
        #     logger.info(f"Saving result to {output} ...")
        #     with open(filepath, "w") as openfile:
        #         openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
        #         progbar = tqdm(total=test_dataset.total_steps, unit="batch")
        #         for i, pred in enumerate(results):
        #             groundtruth, greedy, beamsearch = [x.decode("utf-8") for x in pred]
        #             path, duration, _ = test_dataset.entries[i]
        #             openfile.write(f"{path}\t{duration}\t{groundtruth}\t{greedy}\t{beamsearch}\n")
        #             progbar.update(1)
        #         progbar.close()
        # app_util.evaluate_results(filepath)

        """
        store ppg in the form of .npy files
        """
        if overwrite:
            logger.info(f"Saving result to {config.learning_config.test_dataset_config.ppg_dir} ...")
            progbar = tqdm(total=test_dataset.total_steps, unit="batch")
            for i,batch in enumerate(test_data_loader): ## enable us to save the ppg matrix batch by batch
                results = model.predict(batch, verbose=1)
                for j,ppg in enumerate(results):
                    path,_,_ = test_dataset.entries[i*config.learning_config.running_config.batch_size+j]
                    wav_filename = path.split("/")[-1][:-len(".flac")]
                    Path( config.learning_config.test_dataset_config.ppg_dir).mkdir( parents=True, exist_ok=True )
                    ppg_out_path = config.learning_config.test_dataset_config.ppg_dir+wav_filename+".npy"
                    np.save(ppg_out_path,ppg)
                    progbar.update(1)
            progbar.close()


def convert_tflite(
    model: BaseModel,
    output: str,
):
    concrete_func = model.make_tflite_function().get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    output = file_util.preprocess_paths(output)
    with open(output, "wb") as tflite_out:
        tflite_out.write(tflite_model)
