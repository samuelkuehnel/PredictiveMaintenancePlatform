import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import base64
from websockets.asyncio.server import ServerConnection
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
import ast
from tensorflow.keras.losses import mse, mae  # type: ignore
import time


class Dataset:
    def __init__(self, num_features: int,
                 replacement_mode: str = "remove",
                 replacement_value: float = 0.0,
                 mean_scaler: np.ndarray = None,
                 std_scaler: np.ndarray = None,
                 invalid_data_handling: str = "empty",
                 freeze_scaler: int = 3000):
        self.num_features = num_features
        self.invalid_data_handling_dict = None
        self.dataset = pd.DataFrame()
        self.dataset_raw = pd.DataFrame()
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        if mean_scaler is not None and std_scaler is not None:
            self.scaler.mean_ = mean_scaler
            self.scaler.scale_ = std_scaler
            self.scaler.var_ = std_scaler ** 2
            self.skip_scaler_training = True
            print("Skip scaler training")
        else:
            self.skip_scaler_training = False
            print("Train scaler")
        self.buffer = None
        self.replacement_mode = replacement_mode
        self.replacement_value = replacement_value
        self.set_invalid_data_handling(invalid_data_handling)
        self.freeze_scaler = freeze_scaler

    def set_invalid_data_handling(self, config: str):
        if config == "empty":
            print("Invalid data handling is empty")
            return
        self.invalid_data_handling_dict = json.loads(config)
        # Add default value 0 if not present
        for key in self.invalid_data_handling_dict.keys():
            self.invalid_data_handling_dict[key] = ast.literal_eval(
                self.invalid_data_handling_dict[key])
            if len(self.invalid_data_handling_dict[key]) == 1:
                self.invalid_data_handling_dict[key].append(0.0)

    async def save_dataset(self, websocket: ServerConnection):
        """Send dataset to client

        Args:
            websocket (websockets.asyncio.server.ServerConnection): Websocket to send data to  # noqa: E501
        """
        self.dataset.to_csv("dataset.csv", index=False)
        self.dataset_raw.to_csv("dataset_raw.csv", index=False)
        await self.send_file("dataset.csv", websocket)
        await self.send_file("dataset_raw.csv", websocket)

    @staticmethod
    async def send_file(file_name: str,
                        websocket: ServerConnection):
        """Reads files and sends them to the client as bay64 encoded strings

        Args:
            file_name (str): Name of the file to send
            websocket (websockets.asyncio.server.ServerConnection): Websocket server to send data to  # noqa: E501
        """
        with open(file_name, "rb") as f:
            encoded_binary = base64.b64encode(f.read()) \
                                            .decode('utf-8')
            message = "FILE:" + file_name + ":::" + encoded_binary
            if len(message.encode('utf-8')) > 1048000:
                times = int(len(message.encode('utf-8')) / 1048000) + 1
                messages = Dataset.split_string(message, times)
                for chunk in messages:
                    await websocket.send("CHUNK:" + chunk)
                await websocket.send("CHUNKS_COMPLETE")
            else:
                await websocket.send(message)

    @staticmethod
    def split_string(s: str, n: int) -> list:
        chunk_size = len(s) // n
        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = len(s) if i == n - 1 else start + chunk_size
            chunks.append(s[start:end])
        return chunks

    @staticmethod
    def calculate_error(data: np.ndarray,
                        reconstructed: np.ndarray,
                        metric: str):
        """Caluculate the reconstruction error between data and reconstructed
           data

        Args:
            data (np.ndarray): original data
            reconstructed (np.ndarray): reconstructed data
            metric (str): metric to calculate the error
        """
        if metric == "mse":
            # Use metric from model to evaluate the error
            loss = mse(data, np.squeeze(reconstructed)).numpy()
            return loss
        elif metric == "mae":
            loss = mae(data, np.squeeze(reconstructed)).numpy()
            return loss
        elif metric == "mahalanobis":
            cov_matrix = np.cov(np.vstack([data, reconstructed]), rowvar=False)
            inv_cov_matrix = None
            if np.linalg.det(cov_matrix) == 0:
                # Covariance matrix is singular -> use pseudo inverse
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
            else:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            return mahalanobis(data, reconstructed, inv_cov_matrix)
        elif metric == 'wasserstein':
            return wasserstein_distance(data, reconstructed)
        else:
            raise ValueError("Metric not supported")

    def read_data_string(self, json_str: str):
        """Convert json string to pandas DataFrame and scales it

        Args:
            json_str (str): New data as json string

        Returns:
            np.ndarray: New data as scaled numpy array
        """
        skip_sample = False
        data_dict = json.loads(json_str)
        # convert to DataFrame to convert and preprocess data
        frame = pd.DataFrame([data_dict])
        # restore dataset order
        if len(self.dataset_raw) > 0:
            frame = frame[self.dataset_raw.columns]
        try:
            frame = frame.drop(columns=['time'])
        except KeyError:
            pass

        def func(x):
            return pd.to_numeric(x, errors='coerce')
        frame = frame.apply(func, axis=0)
        # frame = frame.astype(float)

        for i in range(len(frame.columns)):
            feature = frame.columns[i]
            value = frame[feature].values[0]
            value = self.filter_feature(value, feature, i)
            frame.at[0, feature] = value
            if value is None:
                skip_sample = True
        data_array = frame.to_numpy()
        if skip_sample:
            # Skip sample if any value is None
            transformed_array = np.empty((0, self.num_features))
        else:
            iterations = len(self.dataset_raw)
            if not self.skip_scaler_training \
                    and iterations < self.freeze_scaler:
                # Train scaler on new data
                self.scaler.partial_fit(data_array)
            transformed_array = self.scaler.transform(data_array)

        # Save data to dataset
        new_data = pd.DataFrame(transformed_array)
        new_data['time'] = time.time_ns()
        self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)
        self.dataset_raw = pd.concat([self.dataset_raw, frame],
                                     ignore_index=True)
        # Return nunmpy array of values
        values_array = transformed_array
        # values_array = self.filter_data(values_array)
        return values_array

    def filter_feature(self, data: float, feature: str, index: int) -> float:
        config = None
        try:
            config = self.invalid_data_handling_dict.get(feature, None)
        except Exception:
            pass
        if config is None:
            # use default config
            config = [self.replacement_mode, self.replacement_value]
        if not pd.isna(data) and not pd.isnull(data) and not np.isinf(data):
            # abort if data is valid
            return data
        # select new value
        replacement_val = None
        if config[0] == "custom":
            replacement_val = config[1]
        elif config[0] == "mean":
            replacement_val = self.scaler.mean_[index]
        elif config[0] == "forwardfill":
            # use last value in dataset
            try:
                replacement_val = self.dataset_raw[feature].iloc[-1]
            except Exception:
                replacement_val = config[1]
        elif config[0] == "remove":
            # remove sample
            replacement_val = None
        else:
            # use default value
            replacement_val = config[1]
        if pd.isna(replacement_val) or pd.isnull(replacement_val) \
                or np.isinf(replacement_val):
            # use default value if replacement value is invalid
            replacement_val = config[1]
        return replacement_val

    def filter_data(self, data: float) -> float:
        """Replace all NaN or Inf values in the data with
           a specified replacement value

        Args:
            data (np.ndarray): Input data array

        Returns:
            np.ndarray: Data array with NaN and Inf values replaced
        """
        if self.replacement_mode == "replace":
            data = np.where(np.isnan(data) | np.isinf(data),
                            self.replacement_value, data)
        elif self.replacement_mode == "remove":
            data = np.empty((0, data.shape[0]))
        elif self.replacement_mode == "mean":
            mask = np.isnan(data) | np.isinf(data)
            try:
                data[mask] = self.scaler.mean_[mask]
            except AttributeError:
                data[mask] = self.replacement_value
        elif self.replacement_mode == "forwardfill":
            last_value = self.dataset_raw.tail(1)
            mask = np.isnan(data) | np.isinf(data)
            if len(last_value) != 0:
                data[mask] = last_value[0][mask]
            else:
                data[mask] = self.replacement_value
        return data

    def save_to_buffer(self, new_data: np.ndarray):
        pass

    def reset_buffer(self):
        self.buffer = None


class StandardDataset(Dataset):
    """Class to handle and buffer a standard scaled training dataset.
    """
    def __init__(self, num_features: int, max_len_buffer: int = 1,
                 buffer_mode: str = "replace", validation_mode: bool = False,
                 replacement_mode: str = "replace",
                 replacement_value: float = 0.0,
                 mean_scaler: float = None,
                 std_scaler: float = None,
                 invalid_data_handling: str = "empty"):
        """Initialize the dataset

        Args:
            num_features (int): Number of input features per sample
            max_len_buffer (int, optional): Number of samples to store inside
                                            the dataset. Defaults to 1.
            buffer_mode (str, optional): Mode on how the buffer is changed
                                         when new samples arrive.
                                         Defaults to "replace".
                            "replace": When max_len_buffer is reached,
                                       the buffer is deleted and filled again.
                            "append": When max_len_buffer is reached,
                                      the oldest sample is removed and
                                      the new sample is appended.
            replacement_mode (str, optional): Mode on how to handle NaN or Inf
                                              values in the data. Defaults to
                                              "replace".
                            "replace": Replace NaN or Inf values with a
                                       specified value
                            "remove": Remove samples with NaN or Inf values
            replacement_value (float, optional): Value to replace NaN or Inf
                                                 values with. Defaults to 0.0.
            validation_mode (bool, optional): If True, the dataset is used as
                                              validation dataset.
                                              Defaults to False.
        """
        super(StandardDataset, self).__init__(num_features, replacement_mode,
                                              replacement_value, mean_scaler,
                                              std_scaler,
                                              invalid_data_handling)
        self.buffer = np.empty((0, num_features))
        self.max_len_buffer = max_len_buffer
        self.buffer_size = 0
        if buffer_mode not in ["replace", "append"]:
            self.buffer_mode = "replace"
        else:
            self.buffer_mode = buffer_mode
        self.validation_mode = validation_mode
        self.total_buffer_units = max_len_buffer

    def reset_buffer(self):
        self.buffer = np.empty((0, self.num_features))

    def save_to_buffer(self, new_data: np.ndarray):
        """Save new sample to buffer and datasets

        Args:
            new_data (np.ndarray): new sample to save
        """
        if self.buffer_size < self.max_len_buffer or self.validation_mode:
            # Save data to buffer
            self.buffer = np.vstack((self.buffer, new_data))
            self.buffer_size += 1
        else:
            # Buffer is full
            if self.buffer_mode == "replace":
                # Replace buffer
                self.buffer = np.vstack((np.empty((0, self.num_features)),
                                         new_data))
                self.buffer_size = 1
            elif self.buffer_mode == "append":
                # Append new data and remove oldest sample
                self.buffer = np.vstack((self.buffer[1:], new_data))

    def ready_to_train(self):
        """Check weather the buffer is ready to be trained on

        Returns:
            bool: True, if buffer is ready to be trained on. False otherwise.
        """
        if (self.buffer_size >= self.max_len_buffer):
            return True
        return False


class TimeSeriesDataset(Dataset):
    """Class to handle and buffer a time series dataset to
       train a RNN autoencoder.
    """
    def __init__(self, num_features: int, len_seq: int = 10,
                 num_seq: int = 10, buffer_mode: str = "replace",
                 validation_mode: bool = False,
                 replacement_mode: str = "replace",
                 replacement_value: float = 0.0,
                 mean_scaler: float = None,
                 std_scaler: float = None,
                 invalid_data_handling: str = "empty"):
        """Initialize the dataset

        Args:
            num_features (int): Number of input features per sample
            len_seq (int, optional): Length of each stored time series.
                                     Defaults to 10.
            num_seq (int, optional): Number of time series to be stored.
                                     Defaults to 10.
            buffer_mode (str, optional): Mode on how the buffer is changed
                                         when new samples arrive.
                                         Defaults to "replace".
                            "replace": When max_len_buffer is reached,
                                       the buffer is deleted and filled again.
                            "append": When max_len_buffer is reached,
                                      the oldest sample is removed and
                                      the new sample is appended.
            validation_mode (bool, optional): If True, the dataset is used
                                              as validation dataset.
                                              Defaults to False.
            replacement_mode (str, optional): Mode on how to handle NaN or Inf
                                              values in the data. Defaults to
                                                "replace".
                            "replace": Replace NaN or Inf values with a
                                       specified value
                            "remove": Remove samples with NaN or Inf values
            replacement_value (float, optional): Value to replace NaN or Inf
                                                 values with. Defaults to 0.0.
        """
        super(TimeSeriesDataset, self).__init__(num_features, replacement_mode,
                                                replacement_value, mean_scaler,
                                                std_scaler,
                                                invalid_data_handling)
        self.sequence = np.empty((0, num_features))
        self.len_seq = len_seq
        self.num_seq = num_seq
        if buffer_mode not in ["replace", "append"]:
            self.buffer_mode = "replace"
        else:
            self.buffer_mode = buffer_mode
        self.validation_mode = validation_mode
        self.total_buffer_units = num_seq * len_seq

    def save_to_buffer(self, sample: np.ndarray):
        """Add new sample to current time series and create sequences

        Args:
            sample (np.ndarray): New sample to add to the time series
        """
        self.sequence = np.vstack((self.sequence, sample))
        if (len(self.sequence) >= self.len_seq):
            # Sequence is full
            if self.buffer is None:
                # Save first sequence
                self.buffer = np.array([self.sequence])
            elif len(self.buffer) >= self.num_seq and not self.validation_mode:
                if self.buffer_mode == "append":
                    self.buffer = np.append(self.buffer[1:],
                                            np.array([self.sequence]),
                                            axis=0)
                else:
                    self.buffer = np.array([self.sequence])
            else:
                self.buffer = np.append(self.buffer, np.array([self.sequence]),
                                        axis=0)
            self.sequence = np.empty((0, self.num_features))

    def ready_to_train(self):
        """Check weather the buffer is ready to be trained on

        Returns:
            bool: True, if buffer is ready to be trained on. False otherwise.
        """
        if (self.buffer is not None and len(self.buffer) >= self.num_seq
                and len(self.sequence) == 0):
            return True
        return False


def generate_random_dict():
    keys = ['a', 'b', 'c', 'd', 'e']
    return {key: np.random.random() for key in keys}


test_dict = {
    "a": 1.0,
    "b": np.nan,
    "c": 3.0,
    "d": np.inf,
    "e": 5.0,
}

test_config = {
    "b": ["mean", 0.0],
    "c": ["remove", 0.0],
    "d": ["forwardfill", 0.0],
    "e": ["replace", 0.0],
}

if __name__ == "__main__":
    ts_dataset = TimeSeriesDataset(5, 3, 2, "append", False)
    # ts_dataset = StandardDataset(5, 3, "replace", False,
    #                              mean_scaler=np.array([0.27263442, 0.38498817, 0.48593945, 0.73878911, 0.50864426]),
    #                              std_scaler=np.array([0.19827348, 0.25824963, 0.24024309, 0.21892152, 0.30961319]),
    #                              invalid_data_handling=json.dumps(test_config))
    for i in range(20):
        vals = ts_dataset.read_data_string(str(json.dumps(generate_random_dict())))
        ts_dataset.save_to_buffer(vals)
        print("Buffer: ", ts_dataset.buffer)
        print("ready to train: ", ts_dataset.ready_to_train())
        # print("Buffer size: ", ts_dataset.buffer_size)
        # print("Max buffer size: ", ts_dataset.max_len_buffer)
    print("scaler values: ", ts_dataset.scaler.mean_,
          ts_dataset.scaler.scale_)
    print("buffer shape: ", ts_dataset.buffer.shape)
    test_vals = ts_dataset.read_data_string(str(json.dumps(test_dict)))
    print("Test values: ", test_vals)
    ts_dataset.save_to_buffer(test_vals)
    print("dataset: ", ts_dataset.dataset)
    print("last timestamps: ", ts_dataset.dataset['time'].tail(3).to_numpy())


    
    # print(Dataset.calculate_error(
    #     np.array([[1.431756877848297,1.4317568778604082,1.4317568778476624,1.431756877848553,1.4317568778485483,1.4317568778484544,0.8995994542226459],[0.52153957,0.5227131,0.523843,0.5232731,0.52157086,0.5223287,-0.12621874]]),
    #     np.array([0.52153957,0.5227131,0.523843,0.5232731,0.52157086,0.5223287,-0.12621874]),
    #     "mahalanobis"))
    # # Definiere die Vektoren
    vector1 = np.array([1.431756877848297, 1.4317568778604082, 1.4317568778476624, 1.431756877848553, 1.4317568778485483, 1.4317568778484544, 0.8995994542226459])
    vector2 = np.array([0.52153957, 0.5227131, 0.523843, 0.5232731, 0.52157086, 0.5223287, -0.12621874])

    # # Kombiniere die Vektoren zu einer Matrix
    # matrix = np.vstack([vector1, vector2])

    # # Berechne die Kovarianzmatrix
    # cov_matrix = np.cov(matrix, rowvar=False)

    # # Berechne die Pseudo-Inverse der Kovarianzmatrix
    # pseudo_inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # # Berechne die Mahalanobis-Distanz
    # mahalanobis_distance = mahalanobis(vector1, vector2, pseudo_inv_cov_matrix)
    print(wasserstein_distance(vector1, vector2))

    # print(f"Die Mahalanobis-Distanz zwischen den Vektoren beträgt {mahalanobis_distance}.")
    # dataset = StandardDataset(3, 3, replacement_mode="replace", replacement_value="1.0")
    # result = dataset.read_data_string('{"a": "1", "b": 1, "c": 1}')
    # dataset.save_to_buffer(result)
    # print(dataset.buffer)
