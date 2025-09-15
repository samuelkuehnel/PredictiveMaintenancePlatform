import asyncio
import websockets
import signal
import os
import websockets.asyncio
import websockets.asyncio.server
import traceback
from Runner import TrainingRunner
import json

runner = None
msg = ""


async def keepalive_ping(websocket, interval):
    while True:
        await asyncio.sleep(interval)
        try:
            await websocket.ping()
            print("Ping gesendet")
            await websocket.pong()
            print("Pong empfangen")
        except websockets.ConnectionClosed:
            print("Verbindung geschlossen")
            break


async def echo(websocket, autoencoder_type: str = "lstm",
               dropout_rate: float = 0.2, len_buffer: int = 32,
               l2_lambda: float = 0.001, len_seq: int = 1,
               batch_size: int = 1, epochs: int = 1,
               buffer_mode: str = "replace",
               validation_split: float = 0.1,
               distance_metric: str = "mse",
               replace_m: str = "replace",
               replace_v: float = 1,
               invalid_data_handling: str = "empty",
               path=None):
    # asyncio.create_task(keepalive_ping(websocket, 10))
    global runner
    global msg
    if runner is None:
        print("Training runner is None")
        runner = TrainingRunner(websocket=websocket,
                                autoencoder_type=autoencoder_type,
                                dropout_rate=dropout_rate,
                                len_buffer=len_buffer,
                                l2_lambda=l2_lambda,
                                len_seq=len_seq,
                                batch_size=batch_size,
                                epochs=epochs,
                                buffer_mode=buffer_mode,
                                validation_split=validation_split,
                                distance_metric=distance_metric,
                                replace_v=replace_v,
                                replace_m=replace_m,
                                invalid_data_handling=invalid_data_handling)
        print("New training runner created")
    try:
        async for message in websocket:
            # await websocket.send(message + " recieved")
            if str(message).startswith("CHUNK:"):
                msg += str(message).replace("CHUNK:", "")
                print("Chunk received")
            if str(message).startswith("CHUNKS_COMPLETE"):
                message = msg
                print("Chunks complete")
                with open("received_message.json", "w", encoding="utf-8") as f:
                    json.dump(message, f)
                msg = ""
            if str(message).startswith("DATA:"):
                # Training data recieved
                message = str(message).replace("DATA:", "")  # remove DATA:
                runner.websocket = websocket
                runner.train_sample(message)
            elif str(message).startswith("ping"):
                runner.websocket = websocket
                await websocket.send("pong")

            elif str(message).startswith("SHUTDOWN"):
                # Validate Model
                print(message)
                runner.websocket = websocket
                await runner.finish_training()
                await asyncio.sleep(10)
                print("State: ", runner.successful_finish)
                if runner.successful_finish:
                    print("Training finished successfully")
                    await websocket.close(1002, "TRAINING FINISHED")
                    loop = asyncio.get_running_loop()
                    loop.stop()

            elif str(message).startswith("INIT AUTOENCODER:"):
                if runner.counterIterations == 0:
                    num_features = int(str(message).replace(
                        "INIT AUTOENCODER:", ""))
                    runner.init_autoencoder(num_features)
                else:
                    runner.websocket = websocket
                    print("Reconnect - Model not recreated")
            elif str(message).startswith("LOAD AUTOENCODER:"):
                if runner.counterIterations == 0:
                    message = str(message).replace("LOAD AUTOENCODER:", "")
                    # model_file = base64.b64decode(message).decode("utf-8")
                    runner.load_autoencoder(message)
                else:
                    print("Reconnect - Model not recreated")
    except websockets.exceptions.ConnectionClosedError as e:
        print("Connection closed")
        print(e)
    except Exception as e:
        print("Error: ", e)
        print(traceback.format_exc())
        print("Message: ", message)


async def main(host: str, port: int, autoencoder: str = "standard",
               dropout_rate: float = 0.2, len_buffer: int = 32,
               l2_lambda: float = 0.001, len_seq: int = 1,
               batch_size: int = 1, epochs: int = 1,
               buffer_mode: str = "replace",
               val_split: float = 0.1,
               distance_metric: str = "mse",
               replace_m: str = "replace",
               replace_v: float = 0,
               invalid_data_handling: str = "empty",
               path=None):

    async with websockets.serve(lambda w: echo(w,
                                               autoencoder_type=autoencoder,
                                               dropout_rate=dropout_rate,
                                               len_buffer=len_buffer,
                                               l2_lambda=l2_lambda,
                                               len_seq=len_seq,
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               buffer_mode=buffer_mode,
                                               validation_split=val_split,
                                               distance_metric=distance_metric,
                                               replace_v=replace_v,
                                               replace_m=replace_m,
                                               invalid_data_handling=invalid_data_handling),  # noqa: E501
                                host, port, ping_interval=10,
                                ping_timeout=10):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main("0.0.0.0", 8765))
