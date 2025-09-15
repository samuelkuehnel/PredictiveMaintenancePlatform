import tensorflow as tf
import base64
from Dataset import Dataset
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, LSTM, RepeatVector, GRU  # type: ignore
from tensorflow.keras.layers import TimeDistributed, Dense  # type: ignore
from tensorflow.keras.layers import Dropout, MultiHeadAttention, Attention  # type: ignore
from tensorflow.keras.layers import LayerNormalization  # type: ignore
from tensorflow.keras.layers import MaxPooling1D, UpSampling1D  # type: ignore
from tensorflow.keras.layers import Conv1D, Conv1DTranspose  # type: ignore
# from tensorflow.keras.layers import Attention  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
import pandas as pd
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from websockets.asyncio.server import ServerConnection

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.neural_network import MLPRegressor
# import tensorflow_models  as tfm
import json
import numpy as np

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


class Autoencoder():
    """Class to generate different types of autoencoders. The autoencoder can
       be used to compress data and reconstruct it.
       The following types of autoencoders are supported:
         - default: A simple feedforward autoencoder
         - lstm: An LSTM-based autoencoder
    """
    def __init__(self, shape: tuple, model_type: str, dropout_rate=0.2,
                 l2_lambda=0.0001, loss="mse", compiler="adam",
                 model_path: str = "./autoencoder.keras",
                 model_file: str = "", **kwargs):
        """function to initialize the autoencoder

        Args:
            shape (tuple): 0: number of features,
                           1: number of timesteps (for lstm and conv1d)
            model_type (string): Either "standard", "lstm", "attention"
                                 or "conv"
            dropout_rate (float, optional): Rate for dropout regularization.
                                            Defaults to 0.2.
            l2_lambda (float, optional): Lambda for l2 regularization.
                                         Defaults to 0.001.
            loss (str, optional): Loss function for the autoencoder.
                                    Defaults to "mse".
            compiler (str, optional): Optimizer for the autoencoder.
                                      Defaults to "adam".
            model_path (str, optional): Path to the model file.
                                        Defaults to "".
            model_file (str, optional): Model file as JSON string.
                                        Defaults to "".
        """
        super(Autoencoder, self).__init__(**kwargs)
        self.shape = shape
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.model_type = model_type
        self.loss = loss

        self.metric_values_mse = np.array([])
        self.metric_values_mae = np.array([])
        self.metric_values_mahalanobis = np.array([])
        self.metric_values_wasserstein = np.array([])

        self.mse_performance = None
        self.mahalanobis_performance = None
        self.wasserstein_performance = None

        self.threshold_mse = None
        self.threshold_mahalanobis = None
        self.threshold_wasserstein = None

        self.optimizer = compiler
        self.test_loss = None
        self.recommended_threshold = None
        if (model_type == "standard"):
            self.model = self.default_autoencoder()
        elif (model_type == "lstm"):
            self.model = self.lstm_autoencoder()
        elif (model_type == "file"):
            self.model = self.load_model_from_file(model_file, model_path)
        elif (model_type == "attention"):
            self.model = self.attention_autoencoder()
        elif (model_type == "conv"):
            self.model = self.conv_autoencoder()

    def load_model_from_file(self, model_file: str, model_path: str) -> Model:
        """Load model from file .decode("utf-8").encode("ascii")

        Args:
            model_file (str): JSON representation of the model
            model_path (str): Path to the model file

        Returns:
            tensorflow.keras.models.Model: Loaded model
        """
        # Load and save model properties
        model_dict = json.loads(model_file)
        self.model_type = model_dict["model_type"]
        # convert shape to tuple with ints
        self.shape = tuple(map(int, list(model_dict["shape"])))
        try:
            self.dropout_rate = float(model_dict["dropout_rate"])
            self.l2_lambda = float(model_dict["l2_lambda"])
        except Exception as e:
            print("Error loading dropout_rate or l2_lambda -> using standard values", e)
            self.dropout_rate = 0.2
            self.l2_lambda = 0.001
        self.loss = model_dict["loss"]
        self.optimizer = model_dict["optimizer"]
        # Load model itself
        keras_model = base64.b64decode(model_dict["model"].encode("utf-8"))
        with open(model_path, "wb") as f:
            f.write(keras_model)
        model = tf.keras.models.load_model(model_path,
                                           custom_objects={'DecoderLayer':
                                                           DecoderLayer,
                                                           'EncoderLayer':
                                                           EncoderLayer,
                                                           'PositionalEncoding':
                                                           PositionalEncoding,
                                                           'CustomLambdaLayer':
                                                           CustomLambdaLayer})
        return model

    def calculate_loss(self, eval_data: np.ndarray,
                       predictions: np.ndarray,
                       metric: str = "mse") -> float:
        """Calculate loss for a given dataset and metric
        Args:
            eval_data (np.ndarray): Evaluation dataset to validate the model
            predictions (np.ndarray): Predictions from the model
            metric (str): Metric to use for validation ("mse" or "mahalanobis")
        """
        if metric == "mse":
            losses = Dataset.calculate_error(eval_data,
                                             predictions,
                                             "mse")
            self.metric_values_mse = \
                np.append(self.metric_values_mse,
                          losses[~np.isnan(losses)])
        elif metric == "mae":
            losses = Dataset.calculate_error(eval_data,
                                             predictions,
                                             "mae")
            self.metric_values_mae = \
                np.append(self.metric_values_mae,
                          losses[~np.isnan(losses)])
        elif metric == "mahalanobis":
            predictions = np.squeeze(predictions)
            shape_data = predictions.shape
            eval_shape = eval_data.shape
            if len(shape_data) > 2 or len(eval_shape) > 2:
                predictions = np.reshape(predictions,
                                         (-1, shape_data[2]))
                eval_data = np.reshape(eval_data,
                                       (-1, shape_data[2]))
            losses = np.array([Dataset.calculate_error(x, y, "mahalanobis")
                               for x, y in zip(eval_data, predictions)])
            self.metric_values_mahalanobis = \
                np.append(self.metric_values_mahalanobis,
                          losses[~np.isnan(losses)])
        elif metric == "wasserstein":
            predictions = np.squeeze(predictions)
            shape_data = predictions.shape
            eval_shape = eval_data.shape
            if len(shape_data) > 2:
                predictions = np.reshape(predictions,
                                         (-1, shape_data[2]))
            if len(eval_shape) > 2:
                eval_data = np.reshape(eval_data,
                                       (-1, eval_shape[2]))
            losses = np.array([Dataset.calculate_error(x, y, "wasserstein")
                               for x, y in zip(eval_data, predictions)])
            self.metric_values_wasserstein = \
                np.append(self.metric_values_wasserstein,
                          losses[~np.isnan(losses)])
        else:
            raise ValueError("""Unsupported metric. Use 'mse','mahalanobis'
                             or 'wasserstein.""")

    def evaluate_model(self, eval_data: np.ndarray,
                       metric: str = "mse") -> np.ndarray:
        """Evaluate the model with a given dataset
           and return loss
        Args:
            eval_data (np.ndarray): Evaluation dataset to validate the model
            metric (str): Metric to use for validation ("mse" or "mahalanobis")

        Returns:
            np.ndarray: predictions from the model
        """
        predictions = self.model.predict_on_batch(eval_data)
        if metric == "mse":
            self.calculate_loss(eval_data, predictions, "mse")
        elif metric == "mae":
            self.calculate_loss(eval_data, predictions, "mae")
        elif metric == "mahalanobis":
            self.calculate_loss(eval_data, predictions, "mahalanobis")
        elif metric == "wasserstein":
            self.calculate_loss(eval_data, predictions, "wasserstein")
        elif metric == "all":
            self.calculate_loss(eval_data, predictions, "mse")
            #self.calculate_loss(eval_data, predictions, "mahalanobis")
            self.calculate_loss(eval_data, predictions, "wasserstein")
        else:
            raise ValueError("""Unsupported metric. Use 'mse','mahalanobis',
                             'wasserstein' or 'all'.""")
        return predictions

    def get_loss(self, metric: str = "mse", size: int = None,
                 aggregation: str = "max") -> float:
        """Get loss for a given metric
        Args:
            metric (str): Metric to use for validation ("mse" or "mahalanobis")
            size (int, optional): Size of the last loss values to return.
                                    Defaults to None.
            aggregation (str, optional): Aggregation method to use ("mean" or "max").

        Returns:
            float: Mean loss value
            np.ndarray: Losses for each data point
        """
        def aggregate_array(array: np.ndarray, method: str) -> float:
            if method == "mean":
                return float(np.mean(array))
            elif method == "max":
                return float(np.max(array))
            else:
                raise ValueError("Unsupported method. Use 'mean' or 'max'.")
        if size is None:
            size = len(self.metric_values_mse)
        if metric == "mse":
            losses = self.metric_values_mse
            return aggregate_array(losses[-size:], aggregation)
        elif metric == "mahalanobis":
            losses = self.metric_values_mahalanobis
            return aggregate_array(losses[-size:], aggregation)
        elif metric == "wasserstein":
            losses = self.metric_values_wasserstein
            return aggregate_array(losses[-size:], aggregation)
        elif metric == "mae":
            losses = self.metric_values_mae
            return aggregate_array(losses[-size:], aggregation)
        else:
            raise ValueError("""Unsupported metric. Use 'mse', 'mae',
                             'mahalanobis'
                             or 'wasserstein'.""")

    def validate_model(self, val_data: np.ndarray,
                       metric: str = "mse") -> float:
        """Validate the model with a validation dataset
           and save validation loss
        Args:
            val_data (np.ndarray): Validation dataset to validate the model
            metric (str): Metric to use for validation ("mse" or "mahalanobis")

        Returns:
            float: Validation loss
        """
        if val_data.shape[0] == 0:
            print("Validation data is empty.")
            return None
        self.evaluate_model(val_data, "all")
        self.test_loss = float(self.get_loss(metric))
        losses = self.get_losses(metric)
        if len(self.metric_values_mse) > 0:
            quantile = self.generate_threshold(self.metric_values_mse)
            max_value = self.generate_threshold(self.metric_values_mse, "max")
            self.threshold_mse = (max_value, quantile)
        if len(self.metric_values_mae) > 0:
            quantile = self.generate_threshold(self.metric_values_mae)
            max_value = self.generate_threshold(self.metric_values_mae, "max")
            self.threshold_mae = (max_value, quantile)
        if len(self.metric_values_mahalanobis) > 0:
            quantile = self.generate_threshold(self.metric_values_mahalanobis)
            max_value = self.generate_threshold(self.metric_values_mahalanobis,
                                                "max")
            self.threshold_mahalanobis = (max_value, quantile)
        if len(self.metric_values_wasserstein) > 0:
            quantile = self.generate_threshold(self.metric_values_wasserstein)
            max_value = self.generate_threshold(self.metric_values_wasserstein,
                                                "max")
            self.threshold_wasserstein = (max_value, quantile)
        return self.test_loss, losses

    def get_losses(self, metric: str = "mse") -> np.ndarray:
        """Get losses for a given metric
        Args:
            metric (str): Metric to use for validation ("mse" or "mahalanobis")

        Returns:
            np.ndarray: Losses for each data point
        """
        if metric == "mse":
            return self.metric_values_mse
        elif metric == "mahalanobis":
            return self.metric_values_mahalanobis
        elif metric == "wasserstein":
            return self.metric_values_wasserstein
        elif metric == "mae":
            return self.metric_values_mae
        else:
            raise ValueError("""Unsupported metric.
                             Use 'mse','mae','mahalanobis'
                             or 'wasserstein.""")

    def mahalanobis_distance(self, x, y, inv_cov_matrix):
        """Calculate Mahalanobis distance between two points
        Args:
            x (np.ndarray): First point
            y (np.ndarray): Second point
            inv_cov_matrix (np.ndarray): Inverse covariance matrix

        Returns:
            float: Mahalanobis distance
        """
        diff = x - y
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
        return dist

    def conv_autoencoder(self) -> Model:
        # Encoder
        def calc_stride(input_shape: int, kernel_size: int,
                        output_shape: int) -> int:
            """Calculate the stride of a Conv1D layer"""
            return (input_shape - kernel_size + 2*1) // output_shape - 1
        kernel_size = 3
        filter_size = self.shape[0]
        filter_2 = filter_size // 2 if filter_size // 2 > 1 else 2
        filter_4 = filter_size // 4 if filter_size // 4 > 1 else 2
        # stride_1 = calc_stride(filter_size // 4, kernel_size, filter_size // 2)
        # stride_2 = calc_stride(filter_size // 2, kernel_size, filter_size)
        # stride_1 = 1 if stride_1 < 1 else stride_1
        # stride_2 = 1 if stride_2 < 1 else stride_2

        inputs = Input(shape=(self.shape[1], self.shape[0]))
        x = Conv1D(filter_size, kernel_size, activation='relu', padding='same',
                   kernel_regularizer=l2(self.l2_lambda))(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(filter_2, kernel_size, activation='relu', padding='same',
                   kernel_regularizer=l2(self.l2_lambda))(x)
        # x = Dropout(self.dropout_rate)(x)
        # x = MaxPooling1D(1, padding='same')(x)
        encoded = Conv1D(filter_4, kernel_size, activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_lambda))(x)
        # Decoder
        # decoded = Conv1D(filter_4, kernel_size, activation='relu', padding='same',
        #                  kernel_regularizer=l2(self.l2_lambda))(encoded)
        # decoded = UpSampling1D(1)(encoded)
        decoded = Conv1D(filter_2, kernel_size, activation='relu', padding='same',
                                  kernel_regularizer=l2(self.l2_lambda))(encoded)
        decoded = UpSampling1D(2)(decoded)
        # decoded = Conv1D(filter_size, kernel_size, activation='linear', padding='same',
                        #  kernel_regularizer=l2(self.l2_lambda))(decoded)
        # decoded = layers.Cropping1D(cropping=(1, 1))(decoded)  # z. B. von 12 → 10
        decoded = Dropout(self.dropout_rate)(decoded)
        decoded = Conv1D(self.shape[0], kernel_size, activation='linear', padding='same')(decoded)
        # Nach dem Decoder:
        decoded = CustomLambdaLayer(lambda_func, self.shape[1])(decoded)
        # decoded = layers.Lambda(lambda x: x[:, :self.shape[1], :])(decoded)
        return Model(inputs, decoded)

    def attention_autoencoder(self) -> Model:
        """Create attention autoencoder

        Returns:
            tensorflow.keras.models.Model: Default autoencoder
        """
        autoencoder_input = layers.Input(shape=(self.shape[1],
                                                self.shape[0]))
        latent_dim_1 = self.shape[0]
        latent_dim_2 = self.shape[0]//2 if self.shape[0]//2 > 1 else 2
        latent_dim_3 = latent_dim_2
        latent_dim_4 = self.shape[0]//4 if self.shape[0]//4 > 1 else 2
        num_heads = latent_dim_4 // 2 if latent_dim_4 // 2 > 1 else 2
        # x = Dense(latent_dim_1, activation='linear')(autoencoder_input)
        x = PositionalEncoding(self.shape[1], latent_dim_1)(autoencoder_input)
        position_embedding = layers.Embedding(input_dim=self.shape[1],
                                              output_dim=latent_dim_1)
        positions = tf.range(start=0, limit=self.shape[1], delta=1)
        x += position_embedding(positions)
        x = Dense(latent_dim_1, activation='relu')(x)
        x = Dense(latent_dim_2, activation='relu')(x)
        x = Dense(latent_dim_3, activation='relu')(x)
        x = Dense(latent_dim_4, activation='relu')(x)

        attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=latent_dim_4)(x, x)
        x = x + attention_output
        # x = LayerNormalization()(x)

        x = Dense(latent_dim_3, activation='relu')(x)
        x = Dense(latent_dim_2, activation='relu')(x)
        x = Dense(latent_dim_1, activation='relu')(x)
        x = Dense(latent_dim_1, activation='linear')(x)
        model = Model(autoencoder_input, x, name="attention_autoencoder")
        return model

    def default_autoencoder(self) -> Model:
        """Create default autoencoder

        Returns:
            tensorflow.keras.models.Model: Default autoencoder
        """
        latent_dim_1 = self.shape[0]
        latent_dim_2 = self.shape[0]//2 if self.shape[0]//2 > 1 else 2
        latent_dim_3 = latent_dim_2
        latent_dim_4 = self.shape[0]//4 if self.shape[0]//4 > 1 else 2
        encoder = tf.keras.Sequential([
            layers.Dense(latent_dim_1, input_shape=self.shape,
                         activation='relu',
                         kernel_regularizer=l2(self.l2_lambda)),
            Dropout(self.dropout_rate),
            layers.Dense(latent_dim_2, activation='relu',
                         kernel_regularizer=l2(self.l2_lambda)),
            # Dropout(self.dropout_rate),
            layers.Dense(latent_dim_3, activation='relu',
                         kernel_regularizer=l2(self.l2_lambda)),
            # Dropout(self.dropout_rate),
            layers.Dense(latent_dim_4, activation='relu',
                         kernel_regularizer=l2(self.l2_lambda))
        ])
        decoder = tf.keras.Sequential([
            layers.Dense(latent_dim_3, activation='relu',
                         kernel_regularizer=l2(self.l2_lambda)),
            # Dropout(self.dropout_rate),
            layers.Dense(latent_dim_2, activation='relu',
                         kernel_regularizer=l2(self.l2_lambda)),
            # Dropout(self.dropout_rate),
            layers.Dense(latent_dim_1, activation='relu',
                         kernel_regularizer=l2(self.l2_lambda)),
            Dropout(self.dropout_rate),
            layers.Dense(tf.math.reduce_prod(self.shape).numpy(),
                         activation='linear',
                         kernel_regularizer=l2(self.l2_lambda))
        ])
        autoencoder_input = layers.Input(shape=self.shape)
        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)
        autoencoder = Model(autoencoder_input, decoded, name="autoencoder")
        return autoencoder

    def lstm_autoencoder(self) -> Model:
        """Create LSTM autoencoder

        Returns:
            tensorflow.keras.models.Model: LSTM autoencoder
        """
        units = self.shape[0]
        units_2 = int(units // 2) if units // 2 > 1 else 2
        units_4 = int(units // 4) if units // 4 > 1 else 2

        inputs = Input(shape=(self.shape[1], self.shape[0]))
        encoded = LSTM(units, activation='relu',
                       return_sequences=True,
                       kernel_regularizer=l2(self.l2_lambda))(inputs)
        encoded = Dropout(self.dropout_rate)(encoded)
        encoded = LSTM(units_2, activation='relu',
                       return_sequences=True,
                       kernel_regularizer=l2(self.l2_lambda))(encoded)
        # encoded = Dropout(self.dropout_rate)(encoded)
        decoded = LSTM(units_4, activation='relu',
                       return_sequences=True,
                       kernel_regularizer=l2(self.l2_lambda))(encoded)
        # decoded = RepeatVector(self.shape[1])(encoded)
        # decoded = LSTM(units_4, activation='relu',
        #                return_sequences=True,
        #                kernel_regularizer=l2(self.l2_lambda))(encoded)
        # decoded = LSTM(units_2, activation='relu',
        #                return_sequences=True,
        #                kernel_regularizer=l2(self.l2_lambda))(decoded)
        # decoded = Dropout(self.dropout_rate)(decoded)
        # decoded = LSTM(units, activation='relu',
        #                return_sequences=True,
        #                kernel_regularizer=l2(self.l2_lambda))(decoded)
        outputs = TimeDistributed(Dense(self.shape[0],
                                        activation='linear'))(decoded)

        return Model(inputs, outputs)

    def compile_model(self) -> Model:
        """Compile the model and return it

        Returns:
            tensorflow.keras.models.Model: Compiled autoencoder model
        """
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss=self.loss)
        return self.model

    def get_model_base64(self) -> str:
        """save model as base64 encoded string

        Returns:
            str: base64 encoded model
        """
        self.model.save("autoencoder.keras")
        with open("autoencoder.keras", "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def get_model_props(self) -> dict:
        """Return all model properties as dictionary

        Returns:
            dict: Dictionary containing all model properties
        """
        encoded_binary = self.get_model_base64()
        props = {"model_type": self.model_type,
                 "shape": self.shape,
                 "dropout_rate": self.dropout_rate,
                 "l2_lambda": self.l2_lambda,
                 "loss": self.loss,
                 "recommended_threshold_mse": self.threshold_mse,
                 "recommended_threshold_wasserstein":
                 self.threshold_wasserstein,
                 "loss_mse": self.get_loss("mse"),
                 "loss_wasserstein": self.get_loss("wasserstein"),
                 "test_loss": self.test_loss,
                 "optimizer": self.optimizer,
                 "model": encoded_binary}
        return props

    def generate_threshold(self, loss_values: np.ndarray,
                           method: str = "quantile",
                           quantile: float = 0.95) -> float:
        """Generate Recommended Threshold

        Args:
            loss_values (np.ndarray): loss values from detect Phase
            method (str, optional): Method to generate threshold.
                                    Defaults to "quantile".
                                    quantile: Use quantile value as threshold
                                    max: Use max loss as threshold
            quantile (float, optional): Quantile for threshold.
                                        Defaults to 0.95.

        Returns:
            float: Recommended threshold
        """
        if method == "quantile":
            return float(np.quantile(loss_values, quantile))
        elif method == "max":
            return float(np.max(loss_values))
        elif method == "otsu":
            return float(threshold_otsu(loss_values))

    async def save_model(self,
                         websocket: ServerConnection, **kwargs):
        """Saves model with meta information as JSON string

        Args:
            websocket (websockets.asyncio.server.ServerConnection): Websocket server to send data to  # noqa: E501
        """
        json_dict = self.get_model_props()
        # Add additional properties to the JSON dictionary
        for key, value in kwargs.items():
            json_dict[key] = value
        print("dict: ", json_dict)
        json_str = json.dumps(json_dict)
        encoded = base64.b64encode(json_str.encode("ascii")) \
                        .decode('utf-8')
        await websocket.send("FILE:autoencoder.json:::" + encoded)
        model_base64 = self.get_model_base64()
        await websocket.send("FILE:autoencoder.keras:::" + model_base64)

    async def save_losses(self,
                          websocket: ServerConnection):
        """Saves losses as text files

        Args:
            websocket (websockets.asyncio.server.ServerConnection): Websocket server to send data to # noqa: E501
        """
        if self.metric_values_mahalanobis is not None:
            losses = self.metric_values_mahalanobis
            np.savetxt("mahalanobis_losses.txt", losses,
                       delimiter=",")
            await Dataset.send_file("mahalanobis_losses.txt", websocket)
        if self.metric_values_mse is not None:
            losses = self.metric_values_mse
            np.savetxt("mse_losses.txt", losses, delimiter=",")
            await Dataset.send_file("mse_losses.txt", websocket)
        if self.metric_values_wasserstein is not None:
            losses = self.metric_values_wasserstein
            np.savetxt("wasserstein_losses.txt", losses,
                       delimiter=",")
            await Dataset.send_file("wasserstein_losses.txt", websocket)

    def calculate_vram(self) -> float:
        """Calculate VRAM usage of the model

        Returns:
            float: VRAM usage in MB
        """
        total_params = sum([tf.size(v).numpy() for v in
                            self.model.trainable_variables])

        return total_params/1000 * 4 * 1.2


@tf.keras.utils.register_keras_serializable()
class EncoderLayer(layers.Layer):
    def __init__(self, latent_dim_1, latent_dim_2, latent_dim_3,
                 latent_dim_4, l2_lambda, dropout_rate):
        super(EncoderLayer, self).__init__()
        num_heads = latent_dim_4//2
        num_heads = 2 if num_heads < 2 else num_heads
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.latent_dim_3 = latent_dim_3
        self.latent_dim_4 = latent_dim_4
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.dense1 = layers.Dense(latent_dim_1, activation='relu',
                                   kernel_regularizer=l2(l2_lambda))
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(latent_dim_2, activation='relu',
                                   kernel_regularizer=l2(l2_lambda))
        # self.dropout2 = layers.Dropout(dropout_rate)
        self.dense3 = layers.Dense(latent_dim_3, activation='relu',
                                   kernel_regularizer=l2(l2_lambda))
        self.dense4 = layers.Dense(latent_dim_4, activation='relu',
                                   kernel_regularizer=l2(l2_lambda))
        self.attention = MultiHeadAttention(num_heads=num_heads,
                                            key_dim=latent_dim_4)
        # self.attention = Attention()
        self.norm = LayerNormalization()

    def build(self, input_shape):
        num_heads = self.latent_dim_4 // 2
        num_heads = 2 if num_heads < 2 else num_heads
        self.dense1 = layers.Dense(self.latent_dim_1, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(self.latent_dim_2, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        # self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dense3 = layers.Dense(self.latent_dim_3, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        self.dense4 = layers.Dense(self.latent_dim_4, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        self.attention = MultiHeadAttention(num_heads=num_heads,
                                            key_dim=self.latent_dim_4)
        # self.attention = Attention(score_mode="concat")
        self.norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        # x = self.dropout1(x, training=training)
        x = self.dense2(x)
        # x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.norm(x)
        x = tf.expand_dims(x, axis=1)
        x = x + self.attention([x, x])
        x = tf.squeeze(x, axis=1)
        # In order to use Attention layer we need to expand dimensions
        # because Attention layer expects a 3D tensor.
        return x

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config = {
            'latent_dim_1': self.dense1.units,
            'latent_dim_2': self.dense2.units,
            'latent_dim_3': self.dense3.units,
            'l2_lambda': self.dense1.kernel_regularizer.l2,
            'dropout_rate': self.dropout1.rate
        }
        return config

    @classmethod
    def from_config(cls, config):
        new_conf = {}
        new_conf['latent_dim_1'] = config['latent_dim_1']
        new_conf['latent_dim_2'] = config['latent_dim_2']
        new_conf['latent_dim_3'] = config['latent_dim_3']
        new_conf['l2_lambda'] = config['l2_lambda']
        new_conf['dropout_rate'] = config['dropout_rate']
        return cls(**new_conf)


@tf.keras.utils.register_keras_serializable()
class DecoderLayer(layers.Layer):
    def __init__(self, latent_dim_1, latent_dim_2, latent_dim_3, shape,
                 l2_lambda, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.latent_dim_3 = latent_dim_3
        num_heads = latent_dim_3//2
        num_heads = 2 if num_heads < 2 else num_heads
        self.shape = shape
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.dense1 = layers.Dense(latent_dim_3, activation='sigmoid',
                                   kernel_regularizer=l2(l2_lambda))
        self.dense2 = layers.Dense(latent_dim_2, activation='relu',
                                   kernel_regularizer=l2(l2_lambda))
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense3 = layers.Dense(latent_dim_1, activation='relu',
                                   kernel_regularizer=l2(l2_lambda))
        # self.dropout2 = layers.Dropout(dropout_rate)
        # self.attention = MultiHeadAttention(num_heads=num_heads,
        #                                     key_dim=self.latent_dim_3)
        self.attention = Attention()
        self.out = layers.Dense(tf.math.reduce_prod(shape).numpy(),
                                activation='relu',
                                kernel_regularizer=l2(l2_lambda))
        self.norm = layers.LayerNormalization()

    def build(self, input_shape):
        num_heads = self.latent_dim_3//2
        num_heads = 2 if num_heads < 2 else num_heads
        self.dense1 = layers.Dense(self.latent_dim_3, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        self.dense2 = layers.Dense(self.latent_dim_2, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        # self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dense3 = layers.Dense(self.latent_dim_1, activation='relu',
                                   kernel_regularizer=l2(self.l2_lambda))
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.out = layers.Dense(tf.math.reduce_prod(self.shape).numpy(),
                                activation='linear',
                                kernel_regularizer=l2(self.l2_lambda))
        # self.attention = MultiHeadAttention(num_heads=num_heads,
        #                                     key_dim=self.latent_dim_3)
        self.attention = Attention()
        self.norm = layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        x = tf.expand_dims(x, axis=1)
        x = x + self.attention([x, x])
        x = tf.squeeze(x, axis=1)
        x = self.dense1(x)
        # x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        # x = self.dropout2(x, training=training)
        return self.out(x)

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config = {
            'latent_dim_1': self.dense2.units,
            'latent_dim_2': self.dense1.units,
            'latent_dim_3': self.dense3.units,
            'shape': self.out.units,
            'l2_lambda': self.dense1.kernel_regularizer.l2,
            'dropout_rate': self.dropout1.rate
        }
        return config

    @classmethod
    def from_config(cls, config):
        new_conf = {}
        new_conf['latent_dim_1'] = config['latent_dim_1']
        new_conf['latent_dim_2'] = config['latent_dim_2']
        new_conf['shape'] = config['shape']
        new_conf['l2_lambda'] = config['l2_lambda']
        new_conf['dropout_rate'] = config['dropout_rate']
        return cls(**new_conf)


def create_sequences(data, timesteps):
    X = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:i + timesteps])
    return np.array(X)


def batch_fit(data, model, scaler: StandardScaler, batch_size=32, epochs=10):
    """Fit the model in batches

    Args:
        data (np.ndarray): Data to fit the model on
        model (tensorflow.keras.models.Model): Model to fit
        batch_size (int, optional): Batch size for fitting. Defaults to 32.
        epochs (int, optional): Number of epochs to fit. Defaults to 10.
    """
    num_samples = data.shape[0]
    losses = np.array([])
    best_model = model.get_weights()
    for i in range(0, num_samples, batch_size):
        batch_data = np.squeeze(data[i:i + batch_size])
        # print(f"batch_data shape: {batch_data.shape}")
        # scaler.partial_fit(batch_data)
        # batch_data = scaler.transform(batch_data)
        batch_data = batch_data[None, ...]
        print(f"batch_data shape after scaling: {batch_data.shape}")
        if batch_data.shape[0] == batch_size:
            # print(f"batch_data shape: {batch_data.shape} ")
            for epoch in range(epochs):
                loss = model.train_on_batch(batch_data, batch_data)
                losses = np.append(losses, loss)
                # if loss <= np.min(losses):
                #     best_model = model.get_weights()
                print(f"Epoch {epoch + 1}/{epochs}, "
                        f"Batch {i // batch_size + 1}/{num_samples // batch_size + 1}: {loss:.4f}")
            # print(f"Epoch {epoch + 1}/{epochs}, "
            #       f"Batch {i // batch_size + 1}/{num_samples // batch_size + 1}: {loss:.4f}")
    # model.set_weights(best_model)
    # plt.plot(losses, label='Loss training')
    # print(f"Best loss: {np.min(losses):.4f}")
    return model, scaler


def batch_predict(data, model, scaler, batch_size=32):
    """Predict the model in batches

    Args:
        data (np.ndarray): Data to predict
        model (tensorflow.keras.models.Model): Model to predict
        batch_size (int, optional): Batch size for prediction. Defaults to 32.

    Returns:
        np.ndarray: Predictions from the model
    """
    num_samples = data.shape[0]
    predictions = []
    losses = np.array([])
    # prev_batch = None
    # current_batch = None
    for i in range(0, num_samples, batch_size):
        batch_data = np.squeeze(data[i:i + batch_size])
        print(f"batch_data shape: {batch_data.shape}")
        scaler.partial_fit(batch_data)
        batch_data = scaler.transform(batch_data)
        batch_data = batch_data[None, ...]
        # current_batch = batch_data
        # if prev_batch is None:
        #     prev_batch = current_batch
        print(f"batch_data shape after scaling: {batch_data.shape}")
        if batch_data.shape[0] == batch_size:
            preds = model.predict_on_batch(batch_data)
            print(preds)
            mse = Dataset.calculate_error(batch_data,
                                          preds,
                                          "mse")
            losses = np.append(losses, mse)
            predictions.append(preds)
            # prev_batch = current_batch
    plt.plot(losses, label='Loss prediction')
    return np.concatenate(predictions, axis=0)


def test_model():
    model = tf.keras.models.Sequential()

    # Erste Schicht mit 7 Neuronen
    model.add(Dense(7, activation='relu', input_shape=(7,)))

    # Zweite Schicht mit 3 Neuronen
    model.add(Dense(3, activation='relu'))

    # Dritte Schicht mit 3 Neuronen
    model.add(Dense(3, activation='relu'))

    # Vierte Schicht mit 1 Neuron
    model.add(Dense(1, activation='relu'))

    # Fünfte Schicht mit 3 Neuronen
    model.add(Dense(3, activation='relu'))

    # Sechste Schicht mit 3 Neuronen
    model.add(Dense(3, activation='relu'))

    # Siebte Schicht mit 7 Neuronen
    model.add(Dense(7, activation='relu'))

    # Kompiliere das Modell
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def test_attention():
    shape = (10, 7)
    autoencoder_input = layers.Input(shape=shape)
    latent_dim_1 = shape[1]
    latent_dim_2 = shape[1]//2
    latent_dim_3 = shape[1]//2
    latent_dim_4 = shape[1]//2
    x = Dense(latent_dim_1, activation='relu')(autoencoder_input)
    x = PositionalEncoding(10, latent_dim_1)(x)
    position_embedding = layers.Embedding(input_dim=10, output_dim=latent_dim_1)
    positions = tf.range(start=0, limit=10, delta=1)
    x += position_embedding(positions)
    x = Dense(latent_dim_1, activation='relu')(x)
    x = Dense(latent_dim_2, activation='relu')(x)
    x = Dense(latent_dim_3, activation='relu')(x)
    x = Dense(latent_dim_4, activation='relu')(x)
    # pos_enc = tfm.vision.layers.PositionalEncoding(
    #     max_length=10
    # )
    # x = pos_enc(x)
    x = LayerNormalization()(x)
    # x = ExpandLayer()(x)

    # Attention Layer

    attention_output = layers.MultiHeadAttention(num_heads=latent_dim_4//2,
                                                 key_dim=latent_dim_4)(x, x)
    # x = layers.Add()([x, attention_output])
    x = x + attention_output
    # x = SqueezeLayer()(x)
    x = Dense(latent_dim_3, activation='relu')(x)
    x = Dense(latent_dim_2, activation='relu')(x)
    x = Dense(latent_dim_1, activation='relu')(x)
    x = Dense(latent_dim_1, activation='linear')(x)
    model = Model(autoencoder_input, x, name="attention_autoencoder")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


class ExpandLayer(layers.Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=1)


class PosEncLayer(layers.Layer):
    def call(self, x):
        pos_enc = PositionalEncoding(tf.shape(x)[1], tf.shape(x)[2])
        return pos_enc(x)


def lambda_func(x, len=10):
    return x[:, :len, :]


class CustomLambdaLayer(layers.Layer):
    def __init__(self, func, length, **kwargs):
        super(CustomLambdaLayer, self).__init__(**kwargs)
        self.func = func
        self.length = length
        self.func_name = "custom_transform"  # wichtig für get_config

    def call(self, inputs):
        return self.func(inputs, self.length)

    def get_config(self):
        config = super().get_config()
        config.update({
            "custom_transform": self.func_name,
            "length": self.length,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Mapping von Namen zu Funktionen
        func = lambda_func
        length = config["length"]
        return cls(func=func, length=length)


class SqueezeLayer(layers.Layer):
    def call(self, x):
        return tf.squeeze(x, axis=1)


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, d_model):
        super().__init__()
        self.sequence_len = sequence_len
        self.d_model = d_model
        pos = np.arange(sequence_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]

        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config = {
            'sequence_len': self.sequence_len,
            'd_model': self.d_model
        }
        return config

    @classmethod
    def from_config(cls, config):
        new_conf = {}
        new_conf['sequence_len'] = config['sequence_len']
        new_conf['d_model'] = config['d_model']
        return cls(**new_conf)


if __name__ == "__main__":
    # tf.keras.utils.set_random_seed(42)
    # tf.config.experimental.enable_op_determinism()
    # dataset = pd.read_csv("data/datasets/OPCUA Temperature data/dataset_raw_TrainingData.csv")
    dataset = pd.read_csv("data/datasets/pump_sensor_data/sensor_train.csv")
    # dataset = pd.read_csv("data/FSW_example_DATA/cleaned_data/samples_train.csv")
    dataset = dataset.replace([np.nan, np.inf, -np.inf], 0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    # scaler = MinMaxScaler()
    seq_len = 32
    dataset = dataset.to_numpy()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    # dataset = np.squeeze(dataset.to_numpy()
    max_len = dataset.shape[0] // seq_len
    dataset = dataset[:max_len * seq_len, :]
    dataset = dataset.reshape(max_len, seq_len, dataset.shape[1])
    num_features = dataset.shape[2]
    # num_features = dataset.shape[1]
    print(f"dataset shape: {dataset.shape}")
    autoencoder = Autoencoder((num_features,seq_len), loss="mse", model_type="conv", dropout_rate=0.0, l2_lambda=0.0) #conv: dropout: 0.1 l2: 0.0001; -- lstm: dropout: 0,2 seq_len=5, epochs=20;bs=1 -- attention: dropout: 0,2 seq_len=5, epochs=20;bs=2
    # autoencoder = test_model()
    # autoencoder = test_attention()
    # autoencoder = MLPRegressor(hidden_layer_sizes=(7, 3, 3, 1, 3, 3, 7),
    #                            activation='relu', solver='adam',
    #                            batch_size=32)
    # autoencoder.model.build((None, 7))
    autoencoder.compile_model()
    autoencoder.model.summary()
    # autoencoder.fit(dataset, dataset)
    # autoencoder.model.fit(dataset, dataset, epochs=5)
    # first_batch = seqs[:400, :, :]
    # autoencoder.model.fit(dataset, dataset, epochs=20, batch_size=1)
    _, scaler = batch_fit(dataset, autoencoder.model, scaler, epochs=20, batch_size=1)

    # autoencoder.model.save("autoencoder.keras")
    # batch_fit(dataset, autoencoder, batch_size=20)

    # dataset_test = pd.read_csv("data/datasets/OPCUA Temperature data/dataset_raw_standard_validation_2000.csv")
    dataset_test = pd.read_csv("data/datasets/pump_sensor_data/sensor_test_part3_1.csv")
    # dataset_test = pd.read_csv("data/FSW_example_DATA/cleaned_data/samples_val.csv")
    dataset_test = dataset_test.replace([np.nan, np.inf, -np.inf], 0)
    # scaler.fit(dataset_test)
    # dataset_test = scaler.transform(dataset_test.to_numpy())
    dataset_test = np.squeeze(dataset_test.to_numpy())
    max_len_test = dataset_test.shape[0] // seq_len
    dataset_test_sclaed = dataset_test[:max_len_test * seq_len, :]

    seqs = dataset_test_sclaed.reshape(max_len_test, seq_len, dataset_test.shape[1])
    pred = batch_predict(seqs, autoencoder.model, scaler, batch_size=1)
    # pred = autoencoder.model.predict(seqs)
    pred = pred.reshape(max_len_test * seq_len, num_features)
    # print(pred.shape)
    # pred = pd.read_csv("C:\\Users\\KuehnelSamuel\\Downloads\\dataset_predicted.csv", skiprows=1, header=None)
    # scaler = StandardScaler()
    # scaler.mean_ = np.array([71.78226800005756,71.98698613338918,71.7161975999985,71.75774186672483,71.6246000001063,71.7737589333878,71.81275066661146])
    # scaler.scale_ = np.array([0.15985271400746018,0.009554415089114851,0.208359744466962,0.17785911167730414,0.27560812760018966,0.16609983156712693,0.2627922977935239])
    # scaler.var_ = scaler.scale_ ** 2
    print(pred.shape)
    pred = scaler.inverse_transform(pred)
    plt.figure(figsize=(12, 6))
    plt.plot(dataset_test[:, 4], label='Original Data', zorder=0)
    plt.plot(pred[:, 4], label='Reconstructed Data', zorder=0)
    plt.legend()
    plt.show()
    # autoencoder.model.summary()
    # autoencoder.model.save("C:\\Users\\KuehnelSamuel\\Downloads\\autoencoder.keras")
    # # print(autoencoder.test_loss)
    # # print(autoencoder.recommended_threshold)
    # model = tf.keras.models.load_model("C:\\Users\\KuehnelSamuel\\Downloads\\autoencoder.keras",
    #                                    custom_objects={'DecoderLayer':
    #                                                    DecoderLayer,
    #                                                    'EncoderLayer':
    #                                                    EncoderLayer,
    #                                                    'PositionalEncoding':
    #                                                    PositionalEncoding,
    #                                                    'CustomLambdaLayer':
    #                                                    CustomLambdaLayer})
    # model.summary()
