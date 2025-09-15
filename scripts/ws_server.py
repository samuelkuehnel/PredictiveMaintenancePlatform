from ws_server_training import main as run_training
from ws_server_detection import main as run_detection
import os
import asyncio
import tensorflow as tf


tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))


MODE = os.getenv("MODE")
HOST = os.getenv("HOSTNAME")
PORT = int(os.getenv("PORT"))
EPOCHS = int(os.getenv("EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
AUTOENCODER_TYPE = os.getenv("AUTOENCODER_TYPE")
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD"))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE"))
L2_LAMBDA = float(os.getenv("L2_LAMBDA"))
BUFFER_LENGTH = int(os.getenv("BUFFER_LENGTH"))
BUFFER_MODE = os.getenv("BUFFER_MODE")
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH"))
CONTINUE_TRAINING = os.getenv("CONTINUE_TRAINING") == "True"
INCLUDE_ANOMALIES = os.getenv("INCLUDE_ANOMALIES") == "True"
DETECTION_BUFFER_LENGTH = int(os.getenv("DETECTION_BUFFER_LENGTH"))
DETECTION_BUFFER_MODE = os.getenv("DETECTION_BUFFER_MODE")
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC")
REPLACEMENT_MODE = os.getenv("REPLACEMENT_MODE")
REPLACEMENT_VALUE = float(os.getenv("REPLACEMENT_VALUE"))
INVALID_DATA_HANDLING = os.getenv("INVALID_DATA_HANDLING")


if __name__ == "__main__":
    print("Starting server")
    print("Mode: ", MODE)
    if MODE == "TRAINING":
        print("Running training")
        print(f"""Configurations:
              HOST: {HOST}
              PORT: {PORT}
              EPOCHS: {EPOCHS}
              BATCH_SIZE: {BATCH_SIZE}
              AUTOENCODER_TYPE: {AUTOENCODER_TYPE}
              DROPOUT_RATE: {DROPOUT_RATE}
              L2_LAMBDA: {L2_LAMBDA}
              BUFFER_LENGTH: {BUFFER_LENGTH}
              BUFFER_MODE: {BUFFER_MODE}
              SEQUENCE_LENGTH: {SEQUENCE_LENGTH}
              DISTANCE_METRIC: {DISTANCE_METRIC}
              REPLACEMENT_MODE: {REPLACEMENT_MODE}
              REPLACEMENT_VALUE: {REPLACEMENT_VALUE}""")
        asyncio.run(run_training(HOST,
                                 PORT,
                                 autoencoder=AUTOENCODER_TYPE,
                                 dropout_rate=DROPOUT_RATE,
                                 len_buffer=BUFFER_LENGTH,
                                 l2_lambda=L2_LAMBDA,
                                 len_seq=SEQUENCE_LENGTH,
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS,
                                 distance_metric=DISTANCE_METRIC,
                                 replace_m=REPLACEMENT_MODE,
                                 replace_v=REPLACEMENT_VALUE,
                                 invalid_data_handling=INVALID_DATA_HANDLING))
    elif MODE == "DETECTION":
        print("Running detection")
        print(f"""Configurations:
              HOST: {HOST}
              PORT: {PORT}
              DETECTION_THRESHOLD: {DETECTION_THRESHOLD}
              DETECTION_BUFFER_LENGTH: {DETECTION_BUFFER_LENGTH}
              DETECTION_BUFFER_MODE: {DETECTION_BUFFER_MODE}
              CONTINUE_TRAINING: {CONTINUE_TRAINING}
              EPOCHS_TRAINING: {EPOCHS}
              BATCH_SIZE: {BATCH_SIZE}
              BUFFER_LENGTH: {BUFFER_LENGTH}
              SEQUENCE_LENGTH: {SEQUENCE_LENGTH}
              DISTANCE_METRIC: {DISTANCE_METRIC}
              REPLACEMENT_MODE: {REPLACEMENT_MODE}
              REPLACEMENT_VALUE: {REPLACEMENT_VALUE}""")
        asyncio.run(run_detection(HOST, PORT,
                                  len_buffer=DETECTION_BUFFER_LENGTH,
                                  buffer_mode=DETECTION_BUFFER_MODE,
                                  threshold=DETECTION_THRESHOLD,
                                  len_buffer_training=BUFFER_LENGTH,
                                  len_seq_training=SEQUENCE_LENGTH,
                                  buffer_mode_training=BUFFER_MODE,
                                  batch_size_training=BATCH_SIZE,
                                  epochs_training=EPOCHS,
                                  training=CONTINUE_TRAINING,
                                  include_anomalies=INCLUDE_ANOMALIES,
                                  distance_metric=DISTANCE_METRIC,
                                  replace_m=REPLACEMENT_MODE,
                                  replace_v=REPLACEMENT_VALUE,
                                  invalid_data_handling=INVALID_DATA_HANDLING))
    else:
        print("Invalid mode")
        exit(1)
