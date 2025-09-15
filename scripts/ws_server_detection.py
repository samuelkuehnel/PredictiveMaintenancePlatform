import asyncio
import websockets
import numpy as np
import signal
import os
import traceback
from Runner import DetectionRunner
import json


# tf.keras.utils.get_custom_objects().clear()
runner = None
msg = ""


async def keepalive_ping(websocket, interval):
    while True:
        await asyncio.sleep(interval)
        try:
            await websocket.ping()
            print("Ping gesendet")
        except websockets.ConnectionClosed:
            print("Verbindung geschlossen")
            break


async def echo(websocket,
               path_model: str = "./autoencoder.keras",
               threshold: float = 0.8, len_buffer: int = 1,
               len_buffer_training: int = 32, len_seq_training: int = 1,
               buffer_mode_training: str = "replace",
               buffer_mode: str = "replace", batch_size_training: int = 1,
               epochs_training: int = 1,
               training: bool = False,
               include_anomalies: bool = False,
               validation_split: float = 0.2,
               distance_metric: str = "mse",
               validation_size: int = 500,
               replace_m: str = "replace",
               replace_v: float = 0.0, mean_scaler: np.ndarray = None,
               std_scaler: np.ndarray = None,
               invalid_data_handling: str = "empty",
               path=None):
    # asyncio.create_task(keepalive_ping(websocket, 10))
    global runner
    global msg
    if runner is None:
        print("Detection runner is None")
        runner = DetectionRunner(websocket=websocket,
                                 path_model=path_model,
                                 threshold=threshold,
                                 len_buffer=len_buffer,
                                 len_buffer_training=len_buffer_training,
                                 len_seq_training=len_seq_training,
                                 buffer_mode_training=buffer_mode_training,
                                 buffer_mode=buffer_mode,
                                 batch_size_training=batch_size_training,
                                 epochs_training=epochs_training,
                                 training=training,
                                 include_anomalies=include_anomalies,
                                 validation_split=validation_split,
                                 distance_metric=distance_metric,
                                 replace_v=replace_v,
                                 replace_m=replace_m,
                                 invalid_data_handling=invalid_data_handling)
        print("New detection runner created")

    try:
        async for message in websocket:
            if str(message).startswith("CHUNK:"):
                msg += str(message).replace("CHUNK:", "")
                print("Chunk received")
            if str(message).startswith("CHUNKS_COMPLETE"):
                message = msg
                print("Chunks complete")
                with open("received_message.json", "w", encoding="utf-8") as f:
                    json.dump(message, f)
                msg = ""
            if str(message).startswith("LOAD AUTOENCODER:"):
                print("Load autoencoder")
                if runner.iterations == 0:
                    message = str(message).replace("LOAD AUTOENCODER:", "")
                    runner.load_autoencoder(message)
                else:
                    print("Reconnect - Model not recreated")
            elif str(message).startswith("DATA:"):
                # Data recieved -- Conversion
                message = str(message).replace("DATA:", "")
                runner.websocket = websocket
                await runner.detect_sample(message)
            elif str(message).startswith("SHUTDOWN"):
                print("Shutting down")
                runner.websocket = websocket
                await runner.finish_detection()
                if runner.successful_finish:
                    await websocket.close(1002, "DETECTION FINISHED")
                    loop = asyncio.get_running_loop()
                    loop.stop()

    except websockets.exceptions.ConnectionClosedError as e:
        print("Connection closed")
        print(e)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("Message: ", message)


async def main(host: str, port: int,
               path_model: str = "./autoencoder.keras",
               threshold: float = 0.8, len_buffer: int = 1,
               len_buffer_training: int = 32, len_seq_training: int = 1,
               buffer_mode_training: str = "replace",
               buffer_mode: str = "replace", batch_size_training: int = 1,
               epochs_training: int = 1,
               training: bool = False,
               include_anomalies: bool = False,
               distance_metric: str = "mse",
               replace_m: str = "replace",
               replace_v: float = 0, mean_scaler: np.ndarray = None,
               std_scaler: np.ndarray = None,
               invalid_data_handling: str = "empty"):
    async with websockets.serve(lambda ws: echo(ws, path_model=path_model,
                                                threshold=threshold,
                                                len_buffer=len_buffer,
                                                len_buffer_training=len_buffer_training,  # noqa: E501
                                                len_seq_training=len_seq_training,  # noqa: E501
                                                buffer_mode_training=buffer_mode_training,  # noqa: E501
                                                buffer_mode=buffer_mode,  # noqa: E501
                                                batch_size_training=batch_size_training,  # noqa: E501
                                                epochs_training=epochs_training,  # noqa: E501
                                                training=training,
                                                include_anomalies=include_anomalies,  # noqa: E501
                                                distance_metric=distance_metric,  # noqa: E501
                                                replace_v=replace_v,
                                                replace_m=replace_m,
                                                mean_scaler=mean_scaler,
                                                std_scaler=std_scaler,
                                                invalid_data_handling=invalid_data_handling),  # noqa: E501
                                host, port, ping_interval=10, ping_timeout=10):
        await asyncio.Future()  # run infinitely


if __name__ == "__main__":
    asyncio.run(main("0.0.0.0", 8765))
