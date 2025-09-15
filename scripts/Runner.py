from email.mime import message
from websockets.asyncio.server import ServerConnection
import json
import numpy as np
import pandas as pd
from copy import deepcopy
import random
from Autoencoder import Autoencoder
from Dataset import StandardDataset, Dataset, TimeSeriesDataset
import time
import os


class Runner():
    """Parent class for the TrainingRunner and DetectionRunner classes.
    It provides class methods for initializing datasets"""
    @staticmethod
    def create_standard_dataset(num_features: int, len_buffer: int,
                                buffer_mode: str, replace_m: str,
                                replace_v: float,
                                mean_scaler: np.ndarray = None,
                                std_scaler: np.ndarray = None,
                                invalid_data_handling: str = "empty"
                                ) -> StandardDataset:
        """Generates a standard dataset to store and buffer samples.
        This dataset is used for the standard Autoencoder and the
        Attention Autoencoder.
        Args:
            num_features (int): number of input features
            len_buffer (int): number of samples to buffer
            buffer_mode (str): how to update the buffer
            replace_m (str): replacement mode for invalid samples
            replace_v (float): replacement value for invalid samples
            mean_scaler (np.ndarray, optional): mean scaler for the dataset.
                Defaults to None.
            std_scaler (np.ndarray, optional): std scaler for the dataset.
                Defaults to None.
            invalid_data_handling (str, optional): how to handle invalid data.
                Defaults to "empty".
        Returns:
            StandardDataset: StandardDataset object to store samples
        """
        dataset = StandardDataset(num_features,
                                  max_len_buffer=len_buffer,
                                  buffer_mode=buffer_mode,
                                  replacement_mode=replace_m,
                                  replacement_value=replace_v,
                                  mean_scaler=mean_scaler,
                                  std_scaler=std_scaler,
                                  invalid_data_handling=invalid_data_handling)
        return dataset

    @staticmethod
    def create_timeseries_dataset(num_features: int, len_seq: int,
                                  len_buffer: int, replace_m: str,
                                  replace_v: float,
                                  buffer_mode: str = "replace",
                                  mean_scaler: np.ndarray = None,
                                  std_scaler: np.ndarray = None,
                                  invalid_data_handling: str = "empty"
                                  ) -> TimeSeriesDataset:
        """Gernerates a time series dataset to store and buffer samples.
        This dataset is used for the LSTM Autoencoder.

        Args:
            num_features (int): number of input features
            len_seq (int): number of time steps in one sequence
            len_buffer (int): number of sequences to buffer
            replace_m (str): replacement mode for invalid samples
            replace_v (float): replacement value for invalid samples
            buffer_mode (str): how to update the buffer
                               Defaults to "replace".
            mean_scaler (np.ndarray, optional): mean scaler for the dataset.
                Defaults to None.
            std_scaler (np.ndarray, optional): std scaler for the dataset.
                Defaults to None.
            invalid_data_handling (str, optional): how to handle invalid data.
                Defaults to "empty".
        Returns:
            TimeSeriesDataset: TimeSeriesDataset object to store samples
        """
        dataset = TimeSeriesDataset(num_features,
                                    len_seq=len_seq,
                                    num_seq=len_buffer,
                                    buffer_mode=buffer_mode,
                                    replacement_mode=replace_m,
                                    replacement_value=replace_v,
                                    mean_scaler=mean_scaler,
                                    std_scaler=std_scaler,
                                    invalid_data_handling=invalid_data_handling)  # noqa: E501
        return dataset


class TrainingRunner(Runner):
    """Class to run the training process for the Autoencoder model.
    It handles all properties and executes all functions concerned
    with the training process.
    """

    def __init__(self, websocket: ServerConnection,
                 autoencoder_type: str = "lstm",
                 dropout_rate: float = 0.2, len_buffer: int = 32,
                 l2_lambda: float = 0.001, len_seq: int = 1,
                 batch_size: int = 1, epochs: int = 1,
                 buffer_mode: str = "replace",
                 validation_split: float = 0.2,
                 distance_metric: str = "mse",
                 replace_m: str = "replace",
                 replace_v: float = 1,
                 invalid_data_handling: str = "empty"):
        """Initialize the TrainingRunner.
        Args:
            websocket (ServerConnection): websocket connection to the client
            autoencoder_type (str, optional): model type. Defaults to "lstm".
            dropout_rate (float, optional): Dropout rate for model.
                                            Defaults to 0.2.
            len_buffer (int, optional): buffer length for dataset.
                                        Defaults to 32.
            l2_lambda (float, optional): L2 regression strength parameter.
                                         Defaults to 0.001.
            len_seq (int, optional): Sequence length in case of time series.
                                     Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 1.
            epochs (int, optional): Training epochs. Defaults to 1.
            buffer_mode (str, optional): Buffering mode in dataset.
                                         Defaults to "replace".
            validation_split (float, optional): Percentage of validationd data.
                                                Defaults to 0.1.
            distance_metric (str, optional): Metric to calculate
                                             reconstruction error.
                                             Defaults to "mse".
            replace_m (str, optional): Mode to handle invalid data.
                                       Defaults to "replace".
            replace_v (float, optional): Value to replace invalid data.
                                         Defaults to 1.
            invalid_data_handling (str, optional): How to handle invalid data.
                Defaults to "empty".
        """
        super(TrainingRunner, self).__init__()
        self.dataset = None
        self.losses = np.array([])
        self.counterIterations = 0
        self.validationCounter = 0
        self.autoencoder = None
        self.val_dataset = None
        self.websocket = websocket
        self.autoencoder_type = autoencoder_type
        self.dropout_rate = dropout_rate
        self.len_buffer = len_buffer
        self.l2_lambda = l2_lambda
        self.len_seq = len_seq
        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer_mode = buffer_mode
        self.validation_split = validation_split
        self.distance_metric = distance_metric
        self.replace_m = replace_m
        self.replace_v = replace_v
        self.invalid_data_handling = invalid_data_handling
        self.successful_finish = False

    def init_autoencoder(self, num_features: int):
        """Initialize Autoencoder and dataset.
        This function is called when the server receives the
        INIT AUTOENCODER message.
        Args:
            num_features (int): Number of input features
        Raises:
            ValueError: Raised when the model type is invalid
        """
        if self.autoencoder_type == "standard":
            self.autoencoder = Autoencoder((num_features,),
                                           model_type=self.autoencoder_type,
                                           dropout_rate=self.dropout_rate,
                                           l2_lambda=self.l2_lambda)
            self.dataset = self.create_standard_dataset(num_features,
                                                        self.len_buffer,
                                                        self.buffer_mode,
                                                        self.replace_m,
                                                        self.replace_v,
                                                        invalid_data_handling=self.invalid_data_handling)  # noqa: E501
        elif self.autoencoder_type == "lstm" or \
                self.autoencoder_type == "conv" or \
                self.autoencoder_type == "attention":
            self.autoencoder = Autoencoder((num_features, self.len_seq),
                                           model_type=self.autoencoder_type,
                                           dropout_rate=self.dropout_rate,
                                           l2_lambda=self.l2_lambda)
            self.dataset = self.create_timeseries_dataset(num_features,
                                                          self.len_seq,
                                                          self.len_buffer,
                                                          self.replace_m,
                                                          self.replace_v,
                                                          self.buffer_mode,
                                                          invalid_data_handling=self.invalid_data_handling)  # noqa: E501
        else:
            raise ValueError("Unknown model type")
        self.val_dataset = deepcopy(self.dataset)
        self.val_dataset.validation_mode = True
        self.autoencoder.compile_model()
        self.autoencoder.model.summary()
        print("Autoencoder initialized")

    def load_autoencoder(self, message: str):
        """This function is called when the server receives the
        LOAD AUTOENCODER message.
        It loads the Autoencoder model from a file and initializes
        the dataset.
        Args:
            message (str): JSON message containing the model propserties
        Raises:
            ValueError: Raised when the model type is invalid
        """
        self.autoencoder = Autoencoder((0,), "file", model_file=message)
        if self.autoencoder.model_type == 'standard' or \
                self.autoencoder.model_type == 'attention':
            num_features = self.autoencoder.shape[0]
            self.dataset = self.create_standard_dataset(num_features,
                                                        self.len_buffer,
                                                        self.buffer_mode,
                                                        self.replace_m,
                                                        self.replace_v)
        elif self.autoencoder.model_type == 'lstm':
            num_features = self.autoencoder.shape[0]
            len_seq = self.autoencoder.shape[1]
            self.dataset = self.create_timeseries_dataset(num_features,
                                                          len_seq,
                                                          self.len_buffer,
                                                          self.replace_m,
                                                          self.replace_v,
                                                          self.buffer_mode)
        else:
            raise ValueError("Unknown model type")
        self.autoencoder.model.summary()
        self.val_dataset = deepcopy(self.dataset)
        self.val_dataset.validation_mode = True
        self.autoencoder.compile_model()
        print("Autoencoder initialized")

    def train_sample(self, message: str):
        """Adds a raw data sample to the dataset and trains the model.
        This function is called when the server receives the
        DATA message.
        Args:
            message (str): Raw training sample in JSON format
        """
        # Training data recieved -- Conversion
        self.counterIterations += 1
        df = self.dataset.read_data_string(message)
        if self.counterIterations > 500:
            # after 1000 Iterations start to collect validation data
            # Reason: Scaler should be learned properly
            # for validation data
            if random.random() < self.validation_split:
                # With a probability of <validation_split>
                # save data to validation
                self.val_dataset.save_to_buffer(df)
                self.validationCounter += 1
            else:
                self.dataset.save_to_buffer(df)
        else:
            self.dataset.save_to_buffer(df)
        if self.dataset.ready_to_train():
            for _ in range(self.epochs):
                history = self.autoencoder.model.train_on_batch(
                    self.dataset.buffer,
                    self.dataset.buffer)
                # Save loss and metrics history.history["loss"]
                self.losses = np.append(self.losses, history)
                print(f"""Training step: {self.counterIterations}
                      Epoch: {_ + 1}/{self.epochs}
                      Loss: {history}
                      Input shape: {self.dataset.buffer.shape}""")

    async def finish_training(self):
        """Function to send all training information an files to the client.
        It is called when the server receives the TRAINING_FINISHED message.
        """
        # Validate Model
        # allStepsSuccessful = False
        # while not allStepsSuccessful:
        #     allStepsSuccessful = True
        try:
            if self.validationCounter > 0:
                squeezed_data = np.squeeze(self.val_dataset.buffer)
                tup = self.autoencoder.validate_model(squeezed_data,
                                                      self.distance_metric)
                np.savetxt("losses_validation.txt", tup[1],
                           delimiter=",")
                print("Validation saving validation losses")
                await Dataset.send_file("losses_validation.txt",
                                        self.websocket)
            # Training process finished -- Save model and metrics
            means_scaler = self.dataset.scaler.mean_.tolist()
            std_scaler = self.dataset.scaler.scale_.tolist()
            print("Saving model and metrics")
            await self.autoencoder.save_model(self.websocket,
                                              means_scaler=means_scaler,
                                              std_scaler=std_scaler)
            self.successful_finish = True
            # await self.dataset.save_dataset(self.websocket)
            # np.savetxt("losses.csv", np.round(self.losses, decimals=4),
            #            delimiter=",")
            # await Dataset.send_file("losses.csv", self.websocket)
        except Exception as e:
            print("Error: ", e)
            print(self.websocket.close_code)
            self.successful_finish = False
            # allStepsSuccessful = True


class DetectionRunner(Runner):
    """Class to run the training process for the Autoencoder model.
    It handles all properties and executes all functions concerned
    with the training process.
    """
    def __init__(self, websocket: ServerConnection,
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
                 replace_m: str = "replace",
                 replace_v: float = 0,
                 invalid_data_handling: str = "empty"):
        """Class to running the detection process for the Autoencoder model.

        Args:
            websocket (ServerConnection): Websocket connection to the client
            path_model (str, optional): Path to store keras model.
                                    Defaults to "./autoencoder.keras".
            threshold (float, optional): Threshold for anomaly detectopn.
                                    Defaults to 0.8.
            len_buffer (int, optional): Size of the sample buffer.
                                    Defaults to 1.
            len_buffer_training (int, optional): Size of the sample buffer
                                                 used to train the model.
                                                 Defaults to 32.
            len_seq_training (int, optional): Sequence length for retraining
                                            the model.
                                            Defaults to 1.
            buffer_mode_training (str, optional): How buffering should be
                                            handled during retraining.
                                            Defaults to "replace".
            buffer_mode (str, optional): How biffering is done during
                                            detection. Defaults to "replace".
            batch_size_training (int, optional): Mini-Batch size for
                                            retraining. Defaults to 1.
            epochs_training (int, optional): Number of epochs for retraining.
                                        Defaults to 1.
            training (bool, optional): True when the model should be trained
                                    parallel to detection. Defaults to False.
            include_anomalies (bool, optional): True when anomalies should be
                                included in the training. Defaults to False.
            validation_split (float, optional): Pervcentage of samples
                                collected for validation. Defaults to 0.2.
            distance_metric (str, optional): Metric used to caluclate
                                reconstruction error. Defaults to "mse".
            replace_m (str, optional): Mode for dealing with invalid data.
                                        Defaults to "replace".
            replace_v (float, optional): Value to replace invalid data with.
                                        Defaults to 0.
            invalid_data_handling (str, optional): How to handle invalid data.
                    Defaults to "empty".
        """
        self.iterations = 0
        self.successful_finish = False
        self.autoencoder = None
        self.anomalies = {}
        self.losses = np.array([])
        self.dataset_predicted = pd.DataFrame()
        self.dataset = None
        self.dataset_training = None
        self.val_dataset = None
        self.validationCounter = 0
        self.trainingCounter = 0
        self.autoencoder_training = None
        self.model_updated = 0

        self.websocket = websocket
        self.path_model = path_model
        self.threshold = threshold
        self.len_buffer = len_buffer
        self.len_buffer_training = len_buffer_training
        self.len_seq_training = len_seq_training
        self.buffer_mode_training = buffer_mode_training
        self.buffer_mode = buffer_mode
        self.batch_size_training = batch_size_training
        self.epochs_training = epochs_training
        self.training = training
        self.include_anomalies = include_anomalies
        self.validation_split = validation_split
        self.distance_metric = distance_metric
        self.replace_m = replace_m
        self.replace_v = replace_v
        self.invalid_data_handling = invalid_data_handling
        self.states = np.empty((0,2))
        self.last_loss = 0.0

    def load_autoencoder(self, message: str):
        """Load Autoencoder model from file and initialize dataset.

        Args:
            message (str): JSON string containing the model properties
        Raises:
            ValueError: Raised when the model type is invalid
        """
        self.autoencoder = Autoencoder((0,), "file",
                                       model_file=message,
                                       model_path=self.path_model)
        try:
            prop_dict = json.loads(message)
            mean_scaler = np.array(prop_dict["means_scaler"],
                                   dtype=float)
            std_scaler = np.array(prop_dict["std_scaler"],
                                  dtype=float)
            print("Mean scaler: ", mean_scaler)
            print("Std scaler: ", std_scaler)
        except Exception as e:
            print("Error loading scaler: ", e)
            mean_scaler = None
            std_scaler = None
        if self.autoencoder.model_type in ('standard'):
            num_features = self.autoencoder.shape[0]
            self.dataset = self.create_standard_dataset(num_features,
                                                        self.len_buffer,
                                                        self.buffer_mode,
                                                        self.replace_m,
                                                        self.replace_v,
                                                        mean_scaler,
                                                        std_scaler,
                                                        self.invalid_data_handling)   # noqa: E501
            if self.training:
                self.dataset_training = self.create_standard_dataset(
                                                        num_features,
                                                        self.len_buffer,
                                                        self.buffer_mode,
                                                        self.replace_m,
                                                        self.replace_v,
                                                        mean_scaler,
                                                        std_scaler,
                                                        self.invalid_data_handling)   # noqa: E501
        elif self.autoencoder.model_type == 'lstm' or \
                self.autoencoder.model_type == 'conv' or \
                self.autoencoder.model_type == "attention":
            num_features = self.autoencoder.shape[0]
            len_seq = self.autoencoder.shape[1]
            self.dataset = self.create_timeseries_dataset(num_features,
                                                          len_seq,
                                                          self.len_buffer,
                                                          self.replace_m,
                                                          self.replace_v,
                                                          self.buffer_mode,
                                                          mean_scaler,
                                                          std_scaler,
                                                          self.invalid_data_handling)   # noqa: E501
            if self.training:
                self.dataset_training = self.create_timeseries_dataset(
                                                    num_features,
                                                    self.len_seq_training,
                                                    self.len_buffer_training,
                                                    self.replace_m,
                                                    self.replace_v,
                                                    self.buffer_mode_training,
                                                    mean_scaler,
                                                    std_scaler,
                                                    self.invalid_data_handling)   # noqa: E501
        else:
            raise ValueError("Unknown model type")
        if self.training:
            self.val_dataset = deepcopy(self.dataset)
            self.val_dataset.validation_mode = True
            self.autoencoder_training = deepcopy(self.autoencoder)
            self.autoencoder_training.compile_model()
        self.autoencoder.model.summary()
        self.autoencoder.compile_model()

    async def finish_detection(self):
        """Finish detection process and send all information to the client.
        """
        # allStepsSuccessful = False
        # while not allStepsSuccessful:
        # allStepsSuccessful = True
        try:
            json.dump(self.anomalies, open("anomalies.json", "w"),
                      default=str)
            await Dataset.send_file("anomalies.json", self.websocket)
            await self.autoencoder.save_losses(self.websocket)
            print("Number losses: ", self.losses.shape)
            print("Detection counter: ", self.iterations)
            print("Times model updated: ", self.model_updated)
            if self.validationCounter > 0:
                tup = self.autoencoder.validate_model(self.val_dataset.buffer,
                                                      self.distance_metric)
                np.savetxt("losses_validation.txt", tup[1],
                           delimiter=",")
                print("Validation saving validation losses")
                await Dataset.send_file("losses_validation.txt",
                                        self.websocket)
            # await self.dataset.save_dataset(self.websocket)
            self.dataset_predicted.to_csv("dataset_predicted.csv",
                                          index=False)
            await Dataset.send_file("dataset_predicted.csv",
                                    self.websocket)
            # np.savetxt("states.txt", self.states,
            #            delimiter=",", newline="\n")
            # await Dataset.send_file("states.txt", self.websocket)
            if self.training:
                mean_scaler = self.dataset.scaler.mean_.tolist()
                std_scaler = self.dataset.scaler.scale_.tolist()
                # await self.dataset_training.save_dataset(self.websocket)
                await self.autoencoder.save_model(self.websocket,
                                                  means_scaler=mean_scaler,
                                                  std_scaler=std_scaler)
            self.successful_finish = True
        except Exception as e:
            print("Error: ", e)
            self.successful_finish = False
            # allStepsSuccessful = False

    async def check_anomaly(self, loss: float) -> bool:
        """Check if the loss is above the threshold and save the sample
        to a file. Send the file to the client.

        Args:
            loss (float): Reconstruction error of the sample

        Returns:
            bool: True if an anomaly is detected, False otherwise
        """
        if loss > self.threshold:
            self.anomalies[self.iterations] = [pd.Timestamp.now(), loss]
            timestamps = self.dataset.dataset['time'].to_numpy(dtype=np.int64)
            timestamps = timestamps[self.dataset.total_buffer_units:]
            anomaly_array = np.vstack((timestamps, np.ones(len(timestamps)))).T
            # anomaly_values = self.dataset.dataset_raw.tail(1)
            # anomaly_dict = anomaly_values.to_dict(orient='records')[0]  # noqa: E501
            # filename = f"anomaly_{self.iterations}.csv"
            # np.savetxt(filename, anomaly_array, delimiter=",", newline=";")
            # # json.dump(anomaly_dict, open(filename, "w"),
            # #           default=str)
            # await Dataset.send_file(filename,
            #                         self.websocket)
            print("Anomaly detected at iteration: ", self.iterations)
            return True
        return False

    def training_sample(self, message: str, loss_old: float):
        """Trains and updates the model during the detection process.

        Args:
            message (str): Raw training sample in JSON format
            loss_old (float): Loss of the old model
        """
        # Update training dataset
        training_data = self.dataset_training.read_data_string(message)
        if random.random() < self.validation_split:
            # With a probability of <validation_split>
            # save data to validation
            print("Adding sample to validation dataset")
            self.val_dataset.save_to_buffer(training_data)
            self.validationCounter += 1
        else:
            print("Adding sample to training dataset")
            self.dataset_training.save_to_buffer(training_data)
        if self.dataset_training.ready_to_train():
            print("dataset training shape: ",
                  self.dataset_training.buffer.shape)
            # loss_new = self.autoencoder_training.validate_model(
            #     self.dataset_training.buffer,
            #     self.distance_metric)
            reconstructed_data = self.autoencoder_training.evaluate_model(
                self.dataset.buffer, "mse")
            loss_new = self.autoencoder_training.get_loss(
                self.distance_metric,
                self.dataset.total_buffer_units,
                aggregation="mean")
            print(f"""Evaluation:
                    Loss old model: {loss_old}
                    Loss new model: {loss_new}
                    Update Model: {loss_new < loss_old}""")
            # If new model is better than old model
            if loss_new < loss_old:
                weights = self.autoencoder_training.model.get_weights()
                self.autoencoder.model.set_weights(weights)
                print("Model updated")
                self.model_updated += 1
            # Retraing model
            print("Training model: ", self.trainingCounter)
            print("Model updated: ", self.model_updated)
            for _ in range(self.epochs_training):
                print(f"Retraining epoch: {_ + 1}/{self.epochs_training}")
                self.autoencoder_training.model.train_on_batch(
                    self.dataset_training.buffer,
                    self.dataset_training.buffer)
            self.trainingCounter += 1

    async def detect_sample(self, message: str):
        """Detects anomalies in the sample and retrains and updates
        the model if necessary.

        Args:
            message (str): Incoming sample in JSON format
        """
        data = self.dataset.read_data_string(message)
        self.dataset.save_to_buffer(data)
        detected = False
        if self.dataset.ready_to_train():
            size_loss_before = len(
                self.autoencoder.metric_values_mse)
            reconstructed_data = self.autoencoder.evaluate_model(
                self.dataset.buffer, "all")
            size_loss_after = len(
                self.autoencoder.metric_values_mse)
            print("Number of added loss values: ",
                  size_loss_after - size_loss_before)
            shape_data = reconstructed_data.shape
            if len(shape_data) > 2:
                reconstructed_data = np.reshape(reconstructed_data,
                                                (-1, shape_data[2]))
            reconstructed_data = pd.DataFrame(reconstructed_data)
            self.dataset_predicted = pd.concat([self.dataset_predicted,
                                               reconstructed_data],
                                               ignore_index=True)
            loss = self.autoencoder.get_loss(self.distance_metric,
                                             self.dataset.total_buffer_units)
            val_loss = self.autoencoder.get_loss(self.distance_metric,
                                                 self.dataset.total_buffer_units,
                                                 aggregation="mean")
            print("Loss(", self.distance_metric, "): ", loss)
            detected = await self.check_anomaly(loss)
            self.last_loss = val_loss
            await self.send_losses_to_orc(anomaly_detected=detected)
            # Check if current sample should be used for training
        retrain = self.include_anomalies or not detected
        if self.training and retrain:
            # If no anomaly is detected, train the model
            self.training_sample(message, self.last_loss)
        self.iterations += 1

    async def send_losses_to_orc(self, anomaly_detected: bool = False):
        """Send latest losses to shepard.

        Args:
            anomaly_detected (bool, optional): True if an anomaly was detected.
                                               Defaults to False.
        """
        losses = self.autoencoder.get_losses(self.distance_metric)
        units = self.dataset.total_buffer_units
        losses = losses[len(losses) - units:]
        timestamps = self.dataset.dataset['time'].to_numpy(dtype=np.int64)
        timestamps = timestamps[len(timestamps) - units:]
        state = np.zeros(len(timestamps))
        if anomaly_detected:
            state = np.ones(len(timestamps))
        self.states = np.append(self.states, np.vstack((timestamps, state)).T)
        final = np.vstack((timestamps, losses, state)).T
        if os.path.exists("loss_tmp.csv"):
            with open("loss_tmp.csv", "a") as f:
                np.savetxt(f, final, delimiter=",", newline=";")
        else:
            np.savetxt("loss_tmp.csv", final, delimiter=",", newline=";")
        try:
            await Dataset.send_file("loss_tmp.csv", self.websocket)
            os.remove("loss_tmp.csv")
        except Exception as e:
            print("Error sending file:", e)
