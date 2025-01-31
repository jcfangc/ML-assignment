import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Optional, Type
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import dump, load
import os
import base64
import io
import zlib
from enum import Enum, auto
from sklearn.base import BaseEstimator


class CompressedTrainedClassifierBase64(Enum):
    LOGISTIC_REGRESSION = auto()
    DECISION_TREE = auto()
    ENSEMBLE = auto()
    SVM = auto()


BASE64_DICT = {
    CompressedTrainedClassifierBase64.LOGISTIC_REGRESSION: "eJyNV3k81Ov3t2QN0V5CohvKrlKpY29xU2gn0zDDjMbQzEiiSyjRVGKobpLdWJJ1CDmTGUuWjL2kdGnTdqNdyXdud/l+f6/7z+95vc7n85zn9Zzzfp/zPK/P5/WOmJY0U0rix2BrMg/SyEQG3YhGpYvfBP8AEplmRKAF+FKZLKo3hz3z57+mrmRfBpnJpAbQOYkc/eOcYxw9tlwgmU6ksUI4bCmaGYc9jRREpHFi2dKsABpnI5jnGZjsNCOyJe3Eztu/AJV9qCwClc4iM7zJgSzOKbbaPw6B6U0U0/DlOEmylbxpRCaTEEym+lJYHGe2EoNIJwX4E5gsIovMcTJgyzIDaIfJDDEok+hL5LDl/YlHCFRxJs5WkRx7un8QjUUl/EjCYSuSyIEMsrc4ksRhy4mjvAKYZDG0YjCR8SMlgyVmLUsn+AV4MTlbpsSDLU8zJTCILGqAGFyFTvAhE1lB4vrF1Akcp1Vs+R+pxT6HPUscRaN6GdGD/ANDCIFUb3FHOWw15z9cGwaDGLKbQQwMFBP7p23yzCCvv6jJ/IgSs6KTiH/sFW9iyzApxEBxkXInxPMABkkcSlFgy5BYIeJVitbfM/FOKaoFJ/ZUDMeVo+ckzZa04jg7O/+g/+PhJMHieLEViTRaQDDB358YKC5Z/U+WP7AI4m770v3JdBbBK4RFZnKcVIO8pKamJMXn9Mf9kBbbNLHJiE1WbHJiS9om8ddBytF/tJvAoaj/WRZFk6JNWeIkeYJD0aUoUJZSlv1ffpTl/2ZHMTxFMfoDdvrUfwdV+n9xZLwDyD7/QpFzWnXyf3B8LP+fONLilTLVgRjRDFlr4j61t8f4QjSMoV2ViLwOYT2Tg+rGrZiyMG69m1cTSrHJ+ujRikMXKDIeetmYtPWrz4P4AXw885Bz+or78OFJY4oqLwF9vhqvvVb4GBb7Hn5ka50HWW837ilJ5aGl1f4VxiPjaKS7aGz58R4UuYWtI/qPYJhm9c9NUg8wJtfjVj10IrPUMmJtQzumpSQtXl37EPe/CbfXcSnD7b9G7DxgmQgM30B2jVUj3K8PtzTRr4SCZZdiJnfnwBJRw1iichfwump6Wi0vo8HSkuUJY/WoU2nv0XM4EqnhZ7Y212TD/gcdYQyNLGyzv/vY8lkFfOzqW2TzJhPUPQ+trHO/DUfN65pLn5UDv3riWYJuD4Q9mmt1oEQAWZ+j/OZX1IAFbuBdkI8A746hlPMzCoEwp8Iz900W7E1vngjfkAMj39ZVTVbl4uihiaQ3YVzUuW33+4KfBSDnetwr+VMNNq8mtSvuFsCOZcOiG4ISjL4gH1v6OgV33S9ynt15CVfReN0WVgIMHyRjmx3izp67C+4JuTjv6pjzZgIPFU2HWpE7CCXnp/2kU16PTH8j7Yl6Ibh60Ulp/c1gNCqHrxz5GKm8uVon6DqSH80ajGvtRs30eVgY0Asr5NUPPll4C2efKitSCCvBdPkea9KrPoj9uONrjmIdnF9K30F2GkCpEdrHWzJ78dDqZvtPcx/gov7dhsX6abiU9zot7kI9PLkozDHcKULi1Pj4ySc9mGnU4kAfrIfnBxw3bVdJB6XLhDhT0zhQ0fCba+hYCdv1E6vdLhUDr/Uc2zq1GOx8nfSjHPshWhQlfflFBoQUK3hrD7djYYTAR3l6LPYP2hn+7OWDOoUEFYeVVVjA5qpmKpVB94MFxMGtp7HYVi2GpFcB18gHNsoU5WLHmskvHxIrYLGVL3edRDFGLnfode0thcr3E79MzMuC5vm506pf8uCwg56VidslmP21bI2EWimk/v6Yw8y+ggXnbaPDq5NRgePYsmt7PUQNjQQJ98VC9PEH0912iut9QHc77JQOOfNCXxVcEeIZi9Ist/wtcMcKp7u9vYXC/GjLF/wutD5Vkj4s6saRzpkt+6e34Y7Js6YmqxX5d353aI53aQCVog0JinFCeFPQfC6XMQpH5jw60h5SBDrlahF9q2+D0xClN2d2N04mnl60TasXfh0JHv0aeQ8ppzsj7Tf0Y5Df1OU6Uje21vxiY5HTCfRPK6Wv7K0EtxWVw7z9TWDx9b3dJbW7YEZ/PukpMQBbbL1Ixvv7QCVqu7RsRC/M+6x4UhTYCZGe2iuv3euHcdpPS99pJ8NrrW3JogvnQMC4EqF/oBQVc/hfLMb6QVXd8HSKnwi+FRaFupZlo4T59lLXsBrMcvc30V4jhLQB9SfRd5XxqEOr5/X4Mui5llOzTVgNt/McFOcF3MQv9iW1m6zOQjezrmX6T61YNPQ5PvNJEpwY1TRd03ITjCJHcz4NC3GzQf/reTYiPLo0VXnYLxPuamo0ukE5vjJn3pabcxK09lh7ZNjXYu29h4pDSR2YVVwYN0bsxEs19NKGqW5sJ/rLFG0UAYgYU456v+HYulYF+2vV6PdBunJ4bxVsvarlnL03Dhb9+vab34I8OKPxvfHg4nbIOJuqrHZECEcruepLyI2g5OBmIDlbjv/wxCONiiOJ8BtX6VygWyWuXPXIx+mnAbw+Z+hpQlMvPDxNzdbveQ0j+WV72qPuQRhXd8YutQdwZJ/e4/hlnVhAXPY5TLsahiOv7krOLcfJgne7itO6xfku7qaueIwVbYJjY9xxWOt+qUa6sAzyLun3m1bXY9Q3/oAa9yZoDG0zjJvVhem94SHeL9Pg0flp76sL69Bca7m/ET0ehEd0lkDEQTSWf7RYOf4MGrcXBSuYP8W101I2lff1YV3fz2kyGdmwlTmjhFuVivdeNZ24+PtN2OFwNUOv0gasvYe0C41qMdbUrGY+0x3PNm9vlD/Lw1+nCl+m656HserWIiMSD2cNfJ65Sb8IjsTvU+380AlVIQle76wKIG1czmk1rx7O7Sa21M7PBg/N8ksZKQmwtnaj/IH5+TDLkMtzb88Hp/Un2yokbuL92zM2feS1IrVopqPprgZot5PUk9vXBsVD7ou6iY1QqREVEViUAfXZwSWrj1WDhv/nSC36BehDY+ttOtfgtHm5oR23HVSU3LTWqopAamqc+zz/NiyYoeZQq9yPikGvdet3N4GQ6OcSL9UCpaLCJ60FSWi7VDVi7fVGeBxjRhx5Moq0dngh3NoGg9GpLn5u9TBas5a8UNzXg7wqRvkbEVo8rX9uWzEECTWn/Ba6DuPTO6G10U0NIJu01Cxa8ALb+1PT34UipnA7ovxGBeBVxF4Tf7cfX6Vobrm4Mx3jI8bLDLEcF1axnypVirDyovl73Zln0LFFP/X+6zT88ryyZ/PxBty07RXBcksOui7OGGP8KgAXvsfE8NnH6Pec3hMq6IC+G1+WkQIrUW9eVcCVCXc87LJ7X+XaSxAiaG1/H86H488+5QoXNcKW31/uffpbIxTcJvVlfk+Ffv6CQhtRFY4+bunbnD2CHV9/6Vi0pBf2JxnLLfyeh7J8R9/RrDw0S8w6XFveDM9Xxp6cYV2PkocC9WfYc6GgW6/lOL0X9A1mWRaK/zPwtOyeSlszKIeGW6y8kAc8z+2HsgaqINvFppwffxa8QnXDlRpr8T3lp0a7iQz8OtZzsXx5Ego9+m70vuzB8WBJ4owNfZj2xTbUKukO7lm2rkTJTp5P/WW3jkZnLszfpUOJjOrBNo+sWx76IijVv2y7oSwDC1g7Q3n1vUButRm/yewAgxWsBgXOCConx57oS34GCca5MlfF31FlS9GxgeRRbFys2G+vKQCGq0Jand4waDwfETDu9YKqjXwGR6UP2Hq6TUp2zWDcOphJixSAUts0ufnbruP8fp9T1BsiCHl8L9iV3w5RK+0zqXejYbauI2nBHBbq33wZcFGlCCerYrq+BLeAoL/qo+SNAlxipXVZ34OHJBO7MoJsJ7iXrFEJbRLCSZ7e++iqRJg/PtfsVmY+WEzYfCH4XoRj5q4uA8X1aEl8m2m6vAZNVHeM6+ohWI37g7t/OrIrlOqVN5Vhr8Y+5Zq5bah2iiDR+q4MwtdSznzLuwI6llvbDn4XoNGCWKL2gC/wIGaVuizi5TJFz9z1PXjFhe47uakZnzcnVg7MFqLTGu1PpstqweFi8oazjQJYp+05N7G4FB4cK3Lxe5QHk6SEb57cGMj7GGs062k6MLQ9kt0r+SDldqTQ5iAX1j8kec/XrIMOC9u4k+QF1h3v98domdTgJvsXs/eEN6IBr3dq3GQES7vvaApPtuPhXc2rDVLv4qbC8/qZda1YYrHfbsWSO0iIVT4a/+YOnu8yzL82Ogj1g1cCuRt60KYON0cP9uCB7ylH11t3g1OfZwFPohciDw0sbbh+Axi6Cst45umwhk/WSVtUDBGeRs4nNIdhal5XrO6tUhiZ/Rufc68G3tAPGrtQj0NRaGpxgHEwbqC1X2uv4sLUTe8k9RgXwLJgWwdGDXauORVw+nQF5Hq9kXdouovUgRfR77dmwEyF58qOW/xAn//gkLdhBZ4bb8h2XIEot4JQWdsQD0Nnnbv2XOYjd0fdlvLptzCu73zzeZtzcG7HaKLXzhokJTN2KSVXYquJubrlsxv4tumN2oun1ajoLy3/pakOBUdyzF6YxcIAN7dIa2MCfAjqDrtRMACyiygfoi26YdzvTu2W/CZUyvXKMoDf0EDwnf4hrQkPJtzUnIg7BzpVBl1SL/OBKWWwfJvUGTg+p3qe0+0GiBij3XAPFUC3bgJHeLkDFj5cPnBVmA9JJn9rBcX/asl/CYZ/ZMn6f8SA7J96Q9ajny4xW816dKOeq6f7XOuFCenvJKNk+FxIUgmyVeQX+4weS3/WACFDnAX9FpJ8e9bSb10nvmOS5t+4qoS/ZDRBLC5/SGS2jKnRSiMzTpCX0X8ASG+G7A==",
    CompressedTrainedClassifierBase64.DECISION_TREE: None,
    CompressedTrainedClassifierBase64.ENSEMBLE: None,
    CompressedTrainedClassifierBase64.SVM: None,
}


def compress_and_encode_joblib(file_path: str, output_path: str):
    """
    Compress and encode a joblib file to base64.

    Args:
    - file_path (str): Path to the input joblib file.
    - output_path (str): Path to save the base64 encoded output.
    """
    try:
        # Read the joblib file as binary
        with open(file_path, "rb") as infile:
            binary_data = infile.read()

        # Compress the binary data
        compressed_data = zlib.compress(binary_data)

        # Encode the compressed data to base64
        base64_data = base64.b64encode(compressed_data).decode("utf-8")

        # Save the base64 encoded data to a file
        with open(output_path, "w") as outfile:
            outfile.write(base64_data)

        print(f"Base64 encoded and compressed file saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def decode_and_decompress_base64(encoded_base64: str) -> io.BytesIO:
    """
    Decode and decompress a Base64 encoded string.

    Args:
    - encoded_base64 (str): Base64 encoded string to decode and decompress.

    Returns:
    - BytesIO: A stream containing the decompressed binary data.
    """
    if not encoded_base64:
        print("No Base64 data provided.")
        return None

    try:
        # Decode Base64
        compressed_data = base64.b64decode(encoded_base64)

        # Decompress the binary data
        decompressed_data = zlib.decompress(compressed_data)

        # Return as a BytesIO stream
        return io.BytesIO(decompressed_data)
    except Exception as e:
        print(f"An error occurred during decoding and decompression: {e}")
        return None


class DataLoader:
    """
    Data loading and preprocessing.
    """

    def __init__(self):
        self.data: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.train_data: Optional[np.ndarray] = None
        self.train_target: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_target: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None  # Standardization tool

    def load_data(self) -> "DataLoader":
        """
        Load the Covertype dataset.
        """
        dataset = fetch_covtype()
        self.data = dataset.data
        self.target = dataset.target
        print(
            f"Data loaded successfully, data dimensions: {self.data.shape}, target dimensions: {self.target.shape}"
        )
        return self

    def clean_data(self) -> "DataLoader":
        """
        Data cleaning: remove NaN and Inf values.
        """
        if self.data is None:
            raise ValueError(
                "Data not loaded yet. Please call load_data() method first."
            )

        mask = np.all(
            np.isfinite(self.data), axis=1
        )  # Check if each row contains only finite values
        self.data = self.data[mask]
        self.target = self.target[mask]
        print(f"Data cleaning completed, remaining data points: {self.data.shape[0]}")
        return self

    def split_and_scale_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> "DataLoader":
        """
        Split the data into training and test sets, and standardize the split data.

        Parameters:
        - test_size (float): Proportion of the test set, default is 20%.
        - random_state (int): Random seed to ensure reproducibility.
        """
        (
            self.train_data,
            self.test_data,
            self.train_target,
            self.test_target,
        ) = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )
        print(
            f"Data split completed: Training set {self.train_data.shape}, Test set {self.test_data.shape}"
        )

        # Standardize the split data
        self.scaler = StandardScaler()
        self.train_data = self.scaler.fit_transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)
        print("Training and test sets standardization completed.")
        return self


class BaseClassifier:
    """
    Base classifier providing common training, evaluation, and model loading logic.
    """

    def __init__(
        self,
        model: Type[BaseEstimator],
        save_path: str,
        base64_constant: CompressedTrainedClassifierBase64 = None,
    ):
        """
        Initialize the classifier.

        Args:
        - model (BaseEstimator): Classifier model instance.
        - save_path (str): Path to save the model file.
        - base64_constant (CompressedTrainedClassifierBase64): Base64 encoded compressed model constant.
        """
        self.model = model
        self.save_path = save_path
        self.base64_constant = base64_constant

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model, supporting loading pretrained models.

        Args:
        - X (np.ndarray): Training features.
        - y (np.ndarray): Training labels.
        """
        # If a saved model file exists, load it directly
        if os.path.exists(self.save_path):
            self.model = load(self.save_path)
            print(f"Loaded saved model: {self.save_path}")
            return

        # If an embedded Base64 model exists, decode and load it
        if model_stream := decode_and_decompress_base64(
            BASE64_DICT[self.base64_constant]
        ):
            self.model = load(model_stream)
            print("Loaded embedded Base64 model.")
            return

        # Train a new model and save it
        self.model.fit(X, y)
        dump(self.model, self.save_path)
        compress_and_encode_joblib(self.save_path, self.save_path + "_base64.txt")
        print(f"Model training completed and saved to: {self.save_path}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model.

        Args:
        - X (np.ndarray): Test features.
        - y (np.ndarray): Test labels.

        Returns:
        - float: Model accuracy.
        """
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Test set accuracy: {accuracy:.4f}")
        return accuracy


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression Classifier.
    """

    def __init__(self):
        super().__init__(
            model=LogisticRegression(
                penalty="l2",
                solver="saga",
                max_iter=2000,
                tol=3e-4,
                C=1.0,
                fit_intercept=True,
                random_state=42,
                n_jobs=-1,
                verbose=True,
            ),
            save_path="logistic_regression_classifier.joblib",
            base64_constant=CompressedTrainedClassifierBase64.LOGISTIC_REGRESSION,
        )


class DecisionTreeClassifierWrapper(BaseClassifier):
    """
    Decision Tree Classifier.
    """

    def __init__(self):
        super().__init__(
            model=DecisionTreeClassifier(random_state=42),
            save_path="decision_tree_classifier.joblib",
            base64_constant=CompressedTrainedClassifierBase64.DECISION_TREE,
        )


class EnsembleClassifier(BaseClassifier):
    """
    Ensemble Method Classifier (Random Forest).
    """

    def __init__(self):
        super().__init__(
            model=RandomForestClassifier(n_estimators=100, random_state=42),
            save_path="ensemble_classifier.joblib",
            base64_constant=CompressedTrainedClassifierBase64.ENSEMBLE,
        )


class SVMClassifier:
    """
    Support Vector Machine Classifier.
    """

    def __init__(
        self, use_pca: bool = False, n_components: float = 0.95, chunk_size: int = 5000
    ):
        """
        Initialize the Support Vector Machine Classifier.

        Parameters:
        - use_pca (bool): Whether to enable PCA dimensionality reduction.
        - n_components (float): Explained variance ratio for PCA, default is 95%.
        - chunk_size (int): Number of samples per chunk during chunked training, default is 5,000.
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.chunk_size = chunk_size
        self.model = SVC(
            C=1,
            kernel="rbf",
            gamma="scale",
            shrinking=True,
            tol=0.0001,
            cache_size=1000,
            max_iter=-1,
            decision_function_shape="ovr",
            class_weight="balanced",
            verbose=True,  # Enable logging output
            random_state=42,
        )
        self.pca = None  # PCA object
        self.reduced_data = None  # Data after dimensionality reduction

    def apply_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA dimensionality reduction to the data.
        """
        if self.pca is None:
            self.pca = PCA(n_components=self.n_components)
            self.reduced_data = self.pca.fit_transform(X)
            print(
                f"PCA dimensionality reduction completed, reduced to {self.pca.n_components_} dimensions."
            )
        return self.reduced_data

    def partial_train(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train SVM in chunks and return support vectors and support labels.

        Parameters:
        - X (np.ndarray): Input data, shape (n_samples, n_features).
        - y (np.ndarray): Input labels, shape (n_samples,).

        Returns:
        - np.ndarray: Merged support vectors, shape (n_support_vectors, n_features).
        - np.ndarray: Merged support vector labels, shape (n_support_vectors,).
        """
        chunk_indices = np.array_split(np.arange(len(X)), len(X) // self.chunk_size + 1)
        support_vectors = []
        support_labels = []

        for chunk in tqdm(chunk_indices, desc="Training SVM in chunks"):
            X_chunk = X[chunk]
            y_chunk = y[chunk]

            # Check if at least two classes are present
            unique_classes = np.unique(y_chunk)
            if len(unique_classes) < 2:
                print("Skipping this chunk because it contains only one class")
                continue

            # Train on single chunk
            chunk_model = SVC(
                C=1,
                kernel="rbf",
                gamma="scale",
                shrinking=True,
                tol=0.0001,
                cache_size=1000,
                max_iter=-1,
                decision_function_shape="ovr",
                class_weight="balanced",
                random_state=42,
            )
            chunk_model.fit(X_chunk, y_chunk)

            # Save support vectors
            support_vectors.append(chunk_model.support_vectors_)
            support_labels.append(y_chunk[chunk_model.support_])

        # Merge support vectors and labels
        all_support_vectors = np.vstack(support_vectors)
        all_support_labels = np.concatenate(support_labels)

        return all_support_vectors, all_support_labels

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the support vector machine in chunks.

        Parameters:
        - X (np.ndarray): Training data, shape (n_samples, n_features).
        - y (np.ndarray): Training labels, shape (n_samples,).
        """
        # If PCA is enabled, apply dimensionality reduction first
        if self.use_pca:
            X = self.apply_pca(X)

        # Chunked training
        support_vectors, support_labels = self.partial_train(X, y)

        print(f"Total number of support vectors: {support_vectors.shape[0]}")
        print(f"Support vector labels: {support_labels}")

        # Final training
        self.model.fit(support_vectors, support_labels)
        print(f"SVM training completed, using {len(support_vectors)} support vectors.")

    def iterative_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold_ratio: float = 0.01,
        min_vectors: int = 100_000,
        save_path: str = "final_support_vectors.joblib",
    ) -> None:
        """
        Iterative training of SVM based on support vectors, dynamically determining when to stop based on the reduction ratio of support vectors.

        Parameters:
        - X (np.ndarray): Initial training data, shape (n_samples, n_features).
        - y (np.ndarray): Initial training labels, shape (n_samples,).
        - threshold_ratio (float): Threshold ratio for support vector reduction, default is 0.1 (10%).
        - min_vectors (int): Minimum threshold for the number of support vectors; stop iterating if below this value.
        - save_path (str): Path to save the final support vectors and labels.
        """
        # If PCA is enabled, apply dimensionality reduction first
        if self.use_pca:
            X = self.apply_pca(X)

        # Initialize support vectors and labels
        current_vectors = X
        current_labels = y
        iteration = 0
        previous_count = len(current_vectors)

        while len(current_vectors) > min_vectors:

            # Check if iterative result already exists on disk
            if os.path.exists(save_path):
                print(
                    "Detected existing support vectors and labels, loading them for final training."
                )
                self.final_train_from_saved_vectors(save_path)
                return

            print(
                f"Iteration {iteration + 1}, current number of data points: {len(current_vectors)}"
            )

            # Chunked training
            current_vectors, current_labels = self.partial_train(
                current_vectors, current_labels
            )

            # Print logs
            reduction_ratio = (previous_count - len(current_vectors)) / previous_count
            print(
                f"Support vectors reduced by {previous_count - len(current_vectors)} this round ({reduction_ratio:.2%})"
            )

            # Check stop condition
            if reduction_ratio < threshold_ratio:
                print(
                    "Reduction in support vectors below threshold, stopping iteration."
                )
                break

            # Update iteration status
            previous_count = len(current_vectors)
            iteration += 1

        # Save final support vectors and labels
        dump((current_vectors, current_labels), save_path)
        print(f"Final support vectors and labels saved to: {save_path}")

        # Final training
        print(f"Final training, using {len(current_vectors)} support vectors")
        self.model.fit(current_vectors, current_labels)

        # Save final model
        dump(self, "svm_classifier.joblib")
        print("SVM training completed.")

    def final_train_from_saved_vectors(
        self, save_path: str = "final_support_vectors.joblib"
    ) -> None:
        """
        Perform final training from saved support vectors and labels.

        Parameters:
        - save_path (str): Path to saved support vectors and labels.
        """
        # Load support vectors and labels
        support_vectors, support_labels = load(save_path)
        print(f"Loaded {len(support_vectors)} support vectors")

        # Final training
        self.model.fit(support_vectors, support_labels)

        # Save final model
        dump(self, "svm_classifier.joblib")
        print("SVM training completed.")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate SVM performance.

        Parameters:
        - X (np.ndarray): Test data, shape (n_samples, n_features).
        - y (np.ndarray): Test labels, shape (n_samples,).

        Returns:
        - float: Test set accuracy.
        """
        # If PCA is enabled, apply dimensionality reduction first
        if self.use_pca:
            X = self.apply_pca(X)

        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"SVM test set accuracy: {accuracy:.4f}")
        return accuracy


if __name__ == "__main__":

    # Task 6: Split data into 80% training set and 20% test set.
    loader = DataLoader()
    loader.load_data().clean_data().split_and_scale_data()

    # Task 7: Train logistic regression and compute its test set accuracy.
    print("Training logistic regression classifier...")
    logistic_regression = LogisticRegressionClassifier()
    logistic_regression.train(loader.train_data, loader.train_target)
    logistic_regression.evaluate(loader.test_data, loader.test_target)

    # Task 8: Train decision tree and compute its test set accuracy.
    print("Training decision tree classifier...")
    decision_tree = DecisionTreeClassifierWrapper()
    decision_tree.train(loader.train_data, loader.train_target)
    decision_tree.evaluate(loader.test_data, loader.test_target)

    # Task 9: Train ensemble method and compute its test set accuracy.
    print("Training ensemble classifier...")
    ensemble = EnsembleClassifier()
    ensemble.train(loader.train_data, loader.train_target)
    ensemble.evaluate(loader.test_data, loader.test_target)

    # Support Vector Machine Attempt
    print("Training SVM classifier...(Attempt)")
    svm = SVMClassifier()
    svm.iterative_train(loader.train_data, loader.train_target)
    print("Model saved.")
