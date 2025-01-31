from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pymc as pm
from arviz import InferenceData
from pymc.math import dot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib
import os
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from typing import Tuple


class RegressionDataLoader:
    """
    Data loading and preprocessing for regression tasks.
    """

    def __init__(self):
        self.data: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.train_data: Optional[np.ndarray] = None
        self.train_target: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_target: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None  # Standardization tool

    def load_data(
        self, data_path: str = "regression_train.txt"
    ) -> "RegressionDataLoader":
        """
        Load the regression dataset from a text file.

        Parameters:
        - data_path (str): Path to the text file containing the dataset.

        Returns:
        - self: The RegressionDataLoader instance.
        """
        try:
            # Load the data from the file
            data = np.loadtxt(data_path)
            self.data = data[:, 0].reshape(-1, 1)  # X values
            self.target = data[:, 1]  # y values
            print(f"Data loaded successfully, shape: {data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        return self

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42, plot: bool = False
    ) -> "RegressionDataLoader":
        """
        Split the data into training and test sets.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        - random_state (int): Random seed for reproducibility. Default is 42.
        - plot (bool): Whether to plot the data distribution.

        Returns:
        - self: The RegressionDataLoader instance.
        """
        if self.data is None or self.target is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        (
            self.train_data,
            self.test_data,
            self.train_target,
            self.test_target,
        ) = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )
        print(
            f"Data split completed: Training set ({self.train_data.shape[0]} samples), "
            f"Test set ({self.test_data.shape[0]} samples)."
        )

        if plot:
            self.plot_data()

        return self

    def standardize_data(self) -> "RegressionDataLoader":
        """
        Standardize the training and test data using StandardScaler.

        Returns:
        - self: The RegressionDataLoader instance.
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not split. Call split_data() first.")

        self.scaler = StandardScaler()
        self.train_data = self.scaler.fit_transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)
        print("Data standardization completed.")

        return self

    def plot_data(self) -> None:
        """
        Plot the training and test data.
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not split. Call split_data() first.")

        plt.style.use(PLOT_STYLE)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.train_data, self.train_target, label="Training data", alpha=0.7
        )
        plt.scatter(self.test_data, self.test_target, label="Test data", alpha=0.7)
        plt.title("Regression Data: Training vs Test")
        plt.xlabel("X values")
        plt.ylabel("y values")
        plt.legend()
        plt.grid(True)
        plt.savefig("regression_data.png")


class PolynomialRegression:
    """
    Polynomial regression model based on maximum likelihood estimation or least squares fitting.
    """

    def __init__(self, degree: int = 2, alpha: float = 0.1):
        """
        Initialize the polynomial regression model.

        Parameters:
        - degree (int): Degree of the polynomial. Default is 2.
        - alpha (float): Regularization strength for Ridge regression. Default is 0.1.
        """
        self.degree = degree
        self.alpha = alpha
        self.model = make_pipeline(
            PolynomialFeatures(degree=self.degree), Ridge(alpha=self.alpha)
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the polynomial regression model.

        Parameters:
        - X (np.ndarray): Input features, shape (n_samples, n_features).
        - y (np.ndarray): Target values, shape (n_samples,).
        """
        self.model.fit(X, y)
        print(f"Polynomial regression model (degree={self.degree}) training completed.")

    def evaluate(self, X: np.ndarray, y: np.ndarray, plot: bool = False) -> float:
        """
        Test and evaluate the model.

        Parameters:
        - X (np.ndarray): Test set features.
        - y (np.ndarray): Test set target values.
        - plot (bool): Whether to plot the actual vs. predicted values.

        Returns:
        - float: Mean Squared Error (MSE) on the test set.
        """
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        print(f"Test set Mean Squared Error (MSE): {mse:.4f}")

        if plot:
            # Set plot style
            plt.style.use(PLOT_STYLE)
            plt.figure(figsize=(8, 6))

            # Scatter plot
            plt.scatter(y, predictions, alpha=0.7, edgecolor="k", label="Predictions")

            # Ideal diagonal line
            plt.plot(
                [y.min(), y.max()],
                [y.min(), y.max()],
                color="magenta",
                linewidth=2,
                label="Ideal",
            )

            # Axis labels and title
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(
                f"Polynomial Regression (degree={self.degree}): Actual vs Predicted"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(
                f"polynomial_regression_actual_vs_predicted_degree={self.degree}_alpha={self.alpha}.png"
            )

        return mse


class NeuralNetworkRegression(nn.Module):
    """
    Custom neural network regression model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1,
        num_hidden_layers: int = 3,  # Number of hidden layers
        lr: float = 0.01,
    ):
        super(NeuralNetworkRegression, self).__init__()
        self.hidden_size = hidden_size  # Size of hidden layers
        self.num_hidden_layers = num_hidden_layers  # Number of hidden layers
        self.lr = lr  # Learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dynamically build hidden layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList()

        # Input layer to the first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))

        # Intermediate hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Last hidden layer to output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Other components
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss()  # Loss function
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Optimizer

        self.to(self.device)  # Move model to the device

    def forward(self, x):
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        # Final layer
        x = self.output_layer(x)
        return x

    def train_model(
        self,
        X_train: torch.Tensor,  # Training features, shape (n_samples, n_features)
        y_train: torch.Tensor,  # Training targets, shape (n_samples,)
        epochs: int = 100,  # Number of training epochs
        batch_size: int = 32,  # Batch size
    ):
        """
        Train the neural network.

        Parameters:
        - X_train (torch.Tensor): Training features.
        - y_train (torch.Tensor): Training targets.
        - epochs (int): Number of epochs.
        - batch_size (int): Batch size.
        """

        # Define model path
        model_path = f"neural_network_regression_hidden_size={self.hidden_size}_hidden_layers={self.num_hidden_layers}.pth"

        # Load model parameters if file exists
        if os.path.exists(model_path):
            print(
                f"Loading pretrained neural network regression model (hidden size={self.hidden_size}, num of hidden layers={self.num_hidden_layers})"
            )
            state_dict = torch.load(
                model_path, weights_only=True
            )  # Load saved state_dict
            self.load_state_dict(state_dict)  # Load model parameters
            return

        # Data loading
        train_data: TensorDataset = TensorDataset(X_train, y_train)
        # Wrap training data into PyTorch Dataset for loading and batching

        train_loader: DataLoader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        # Use DataLoader for batching; shuffle=True for robustness

        # Set model to training mode
        self.train()

        # Start training iterations
        for epoch in range(epochs):
            # Iterate over batches
            for X_batch, y_batch in train_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()  # Clear gradients

                predictions: torch.Tensor = self(X_batch).squeeze()
                # Forward pass

                loss: torch.Tensor = self.criterion(predictions, y_batch)
                # Compute loss

                loss.backward()  # Backpropagation

                self.optimizer.step()  # Update parameters

            # Print loss every few epochs
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        # Save model parameters
        torch.save(
            self.state_dict(),
            f"neural_network_regression_hidden_size={self.hidden_size}_hidden_layers={self.num_hidden_layers}.pth",
        )
        print(
            f"Neural network regression model training completed. Saved at: {model_path}"
        )

    def evaluate_model(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        plot: bool = False,
    ) -> Tuple[int, int, float]:
        """
        Evaluate the neural network.

        Parameters:
        - X_test (torch.Tensor): Test set features.
        - y_test (torch.Tensor): Test set targets.
        - plot (bool): Whether to plot actual vs. predicted values.

        Returns:
        - Tuple[int, int, float]: (hidden_size, num_hidden_layers, mse)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Move test data to device
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            # Forward pass
            predictions: torch.Tensor = self(X_test).squeeze()
            # Compute MSE
            mse = self.criterion(predictions, y_test).item()
            print(f"Test set Mean Squared Error (MSE): {mse:.4f}")

        # Plot actual vs. predicted values
        if plot:
            predictions_np = predictions.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            # Set plot style
            plt.style.use(PLOT_STYLE)
            plt.figure(figsize=(8, 6))

            # Scatter plot
            plt.scatter(
                y_test_np,
                predictions_np,
                alpha=0.7,
                edgecolor="k",
                label=f"Predictions, MSE={mse:.4f}",
            )

            # Ideal diagonal line
            plt.plot(
                [
                    y_test_np.min(),
                    y_test_np.max(),
                ],
                [
                    y_test_np.min(),
                    y_test_np.max(),
                ],
                color="magenta",
                linewidth=2,
                label="Ideal",
            )

            # Axis labels and title
            plt.xlabel("Actual Values")
            plt.ylabel(f"Predicted Values")
            plt.title(
                f"Actual vs Predicted Values (hidden size={self.hidden_size}, num of hidden layer={self.num_hidden_layers})"
            )

            # Add legend and grid
            plt.legend()
            plt.grid(True)

            # Save plot
            plt.savefig(
                f"neural_network_regression_hidden_size={self.hidden_size}_num_hidden_layers={self.num_hidden_layers}.png"
            )

            plt.close()

        return (self.hidden_size, self.num_hidden_layers, mse)


class BayesianPolynomialRegression:
    """
    Bayesian polynomial regression model.
    """

    def __init__(
        self,
        degree: int = 3,
        tune: int = 1000,
    ):
        self.tune: int = tune
        self.degree = degree  # Highest polynomial degree
        self.trace: InferenceData = None  # To store posterior samples
        self.scaler = StandardScaler()  # For feature standardization
        self.poly = PolynomialFeatures(
            degree=self.degree, include_bias=False
        )  # Polynomial feature expansion

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Perform polynomial expansion and standardization on input features.
        """
        X_poly = self.poly.fit_transform(X)
        return self.scaler.fit_transform(X_poly)

    def train(self, X: np.ndarray, y: np.ndarray, draws: int = 2000) -> None:
        """
        Train the Bayesian regression model using PyMC.

        Parameters:
        - X (np.ndarray): Input features, shape (n_samples, n_features).
        - y (np.ndarray): Target values, shape (n_samples,).
        - draws (int): Number of posterior samples.
        - tune (int): Number of tuning samples for optimizing the sampler.
        """

        if os.path.exists(
            f"bayesian_regression_model_degree={self.degree}_tune={self.tune}.joblib"
        ):
            print(
                f"Loading pretrained Bayesian polynomial regression model (degree={self.degree}, tune={self.tune})"
            )
            model = joblib.load(
                f"bayesian_regression_model_degree={self.degree}_tune={self.tune}.joblib"
            )
            self.trace = model["trace"]
            self.degree = model["degree"]
            self.tune = model["tune"]
            return

        X_transformed = self.transform_features(
            X
        )  # Feature expansion and standardization

        with pm.Model():
            # Set prior distributions
            coefs = pm.Normal(
                "coefs",
                mu=0,
                sigma=10,
                shape=X_transformed.shape[1],  # Shape (n_features,)
            )  # Polynomial coefficients
            intercept = pm.Normal("intercept", mu=0, sigma=10)  # Intercept
            sigma = pm.HalfNormal("sigma", sigma=1)  # Noise standard deviation

            # Polynomial model
            mu = dot(X_transformed, coefs) + intercept

            # Likelihood function
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Posterior sampling
            self.trace = pm.sample(
                draws=draws,
                tune=self.tune,
                return_inferencedata=True,
            )
            print(
                f"Bayesian polynomial regression model training completed. (degree={self.degree}, tune={self.tune})"
            )

            joblib.dump(
                {"trace": self.trace, "degree": self.degree, "tune": self.tune},
                f"bayesian_regression_model_degree={self.degree}_tune={self.tune}.joblib",
            )

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        plot: bool = True,
        X_train: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Evaluate model performance, compute predictions and MSE.

        Parameters:
        - X (np.ndarray): New input features, shape (n_samples, n_features).
        - y (np.ndarray): True target values for error evaluation.
        - plot (bool): Whether to plot predictions vs. ideal line.

        Returns:
        - tuple: (degree, tune, mse)
        """
        if self.trace is None:
            raise ValueError(
                "Model not trained yet. Please call the train method first."
            )

        # Feature expansion and standardization
        try:
            X_transformed = self.scaler.transform(self.poly.transform(X))
        except Exception as e:
            self.transform_features(X_train)
            X_transformed = self.scaler.transform(self.poly.transform(X))

        # Get sampled parameter values from posterior
        coefs_samples = self.trace.posterior["coefs"].values
        intercept_samples = self.trace.posterior["intercept"].values

        # Compute predictions
        predictions = np.einsum("ijk,kl->ijl", coefs_samples, X_transformed.T)
        predictions += intercept_samples[..., None]
        mean_prediction = predictions.mean(axis=(0, 1))  # Average prediction

        # Compute MSE
        mse = mean_squared_error(y, mean_prediction)

        # Visualization
        if plot:
            plt.style.use(PLOT_STYLE)
            plt.figure(figsize=(8, 6))
            plt.scatter(
                y,
                mean_prediction,
                alpha=0.7,
                edgecolor="k",
                label=f"Predictions, MSE={mse:.4f}",
            )
            plt.plot(
                [y.min(), y.max()],
                [y.min(), y.max()],
                color="magenta",
                linewidth=2,
                label="Ideal",
            )
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Bayesian Polynomial Regression: Actual vs Predicted")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                f"bayesian_regression_actual_vs_predicted_degree={self.degree}_tune={self.tune}.png"
            )

            plt.close()

        # Return tuple
        return (self.degree, self.tune, mse)


if __name__ == "__main__":
    PLOT_STYLE = "dark_background"

    # Load data
    data_loader = RegressionDataLoader()
    data_loader.load_data().split_data()

    # ======================================================================================
    # Linear Regression ====================================================================
    # ======================================================================================

    def compare_polynomial_models(
        models: list[PolynomialRegression],
        X: np.ndarray,
        y: np.ndarray,
        save_path: str = "polynomial_comparison.png",
    ) -> list[(int, float, float)]:
        """
        Compare multiple polynomial regression models and plot actual vs. predicted values.

        Parameters:
        - models (list): List of polynomial regression models.
        - X (np.ndarray): Test set features.
        - y (np.ndarray): Test set targets.
        - save_path (str): File path to save the plot. Default is "polynomial_comparison.png".

        Returns:
        - list of tuples: Each tuple contains (degree, alpha, mse).
        """
        plt.style.use(PLOT_STYLE)
        plt.figure(figsize=(12, 24))

        # Color and marker settings
        colors = [
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "yellow",
            "magenta",
            "pink",
        ]
        markers = ["o", "s", "v", "D", "P", "X", "H", "d", "p", "*", "^"]

        performance_data = []
        for i, model in enumerate(models):
            # Predict target values
            predictions = model.model.predict(X)
            mse = mean_squared_error(y, predictions)
            performance_data.append((model.degree, model.alpha, mse))

            # Scatter plot
            plt.scatter(
                y,
                predictions,
                alpha=0.7,
                edgecolor="k",
                label=f"Degree {model.degree}, Alpha {model.alpha}, MSE={mse:.4f}",
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
            )

        # Ideal diagonal line
        plt.plot(
            [y.min(), y.max()],
            [y.min(), y.max()],
            color="magenta",
            linewidth=2,
            label="Ideal",
        )

        # Axis labels and title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Polynomial Regression Models: Actual vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)

        plt.close()

        return performance_data

    def plot_linear_regression_performance(data: list[(int, float, float)]):
        """
        Plot 3D relationship between degree, regularization parameter alpha, and MSE.

        Parameters:
        - data (list[(int, float, float)]): Each element is a tuple (degree, alpha, mse).
        """
        # Separate data
        degrees = [point[0] for point in data]
        alphas = [point[1] for point in data]
        mse_values = [point[2] for point in data]

        # Create grid for surface plot
        degree_grid = np.linspace(min(degrees), max(degrees), 50)
        alpha_grid = np.linspace(min(alphas), max(alphas), 50)
        degree_grid, alpha_grid = np.meshgrid(degree_grid, alpha_grid)

        mse_grid = griddata(
            (degrees, alphas), mse_values, (degree_grid, alpha_grid), method="cubic"
        )

        # Plotting
        plt.style.use(PLOT_STYLE)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        surf = ax.plot_surface(
            degree_grid,
            alpha_grid,
            mse_grid,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )

        # Color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        # Axis labels
        ax.set_xlabel("Degree")
        ax.set_ylabel("Alpha")
        ax.set_zlabel("MSE")
        ax.set_title("Degree-Alpha-MSE Relationship")

        # Save figure
        plt.savefig("3D_Surface_View_of_Linear_Regression_Performance.png")

        plt.close()

    # Select the best model based on performance data
    def select_best_model(
        models: list[PolynomialRegression], performance: list[tuple]
    ) -> PolynomialRegression:
        """
        Select the best-performing polynomial regression model.

        Parameters:
        - models (list[PolynomialRegression]): List of trained models.
        - performance (list[tuple]): Performance data as (degree, alpha, mse).

        Returns:
        - PolynomialRegression: The best model instance.
        """
        # Find index of minimum MSE
        best_index = np.argmin([mse for _, _, mse in performance])
        best_model = models[best_index]
        best_degree, best_alpha, best_mse = performance[best_index]

        print(
            f"Best model: Degree={best_degree}, Alpha={best_alpha}, MSE={best_mse:.4f}"
        )
        return best_model

    # Linear regression
    linear_regression_models = []
    for degree in range(1, 21):
        for alpha in [0.9, 0.5, 0.1, 0.01, 0.001]:
            linear_regression = PolynomialRegression(degree=degree, alpha=alpha)
            linear_regression.train(data_loader.train_data, data_loader.train_target)
            linear_regression.evaluate(data_loader.test_data, data_loader.test_target)
            linear_regression_models.append(linear_regression)

    performance = compare_polynomial_models(
        linear_regression_models, data_loader.test_data, data_loader.test_target
    )

    # Select the best model for regression_test.txt predictions
    best_linear_regression = select_best_model(linear_regression_models, performance)

    plot_linear_regression_performance(performance)

    # ======================================================================================
    # Neural Network Regression ============================================================
    # ======================================================================================

    def compare_nn_models(
        models: list[NeuralNetworkRegression],
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ):
        """
        Compare multiple neural network models and plot predictions.

        Parameters:
        - models (list[NeuralNetworkRegression]): List of trained neural network models.
        - X_test (torch.Tensor): Test set features.
        - y_test (torch.Tensor): Test set targets.
        """
        # Fixed color and marker mappings
        colors = ["blue", "green", "orange", "purple", "cyan"]
        markers = ["o", "s", "D", "^", "v"]

        # Extract hidden sizes and layer counts
        hidden_sizes = sorted(set(model.hidden_size for model in models))
        num_layers = sorted(set(model.num_hidden_layers for model in models))

        # Create color and marker maps
        color_map = {
            size: colors[i % len(colors)] for i, size in enumerate(hidden_sizes)
        }
        marker_map = {
            layers: markers[i % len(markers)] for i, layers in enumerate(num_layers)
        }

        # Set plot style
        plt.style.use(PLOT_STYLE)
        plt.figure(figsize=(8, 6))

        # Move test data to CPU
        y_test_np = y_test.cpu().numpy()

        # Plot ideal diagonal line
        plt.plot(
            [y_test_np.min(), y_test_np.max()],
            [y_test_np.min(), y_test_np.max()],
            color="magenta",
            linewidth=2,
            label="Ideal (Actual Values)",
        )

        # Iterate over models and plot predictions
        for model in models:
            model.eval()
            with torch.no_grad():
                predictions = model(X_test.to(model.device)).squeeze()
                predictions_np = predictions.cpu().numpy()

                # Get color and marker
                color = color_map[model.hidden_size]
                marker = marker_map[model.num_hidden_layers]

                # Plot predictions
                plt.scatter(
                    y_test_np,
                    predictions_np,
                    alpha=0.7,
                    color=color,
                    marker=marker,
                    label=f"Hidden Size={model.hidden_size}, Layers={model.num_hidden_layers}",
                )

        # Axis labels and title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Neural Network Regression: Actual vs Predicted Values")
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig("nn_models_comparison_with_shapes_colors.png")

        plt.close()

    def plot_nn_performance(records: list[(int, int, float)]):
        # Extract hidden sizes, number of layers, and MSE
        hidden_sizes = [record[0] for record in records]
        num_layers = [record[1] for record in records]
        mse_values = [record[2] for record in records]

        # Create grid data
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(hidden_sizes), max(hidden_sizes), 100),
            np.linspace(min(num_layers), max(num_layers), 100),
        )

        # Interpolate data
        grid_z = griddata(
            (hidden_sizes, num_layers), mse_values, (grid_x, grid_y), method="cubic"
        )

        # Compute gradients
        grad_x, grad_y = np.gradient(
            grid_z,
            np.linspace(min(hidden_sizes), max(hidden_sizes), 100),
            np.linspace(min(num_layers), max(num_layers), 100),
        )

        # Create 3D plot
        plt.style.use(PLOT_STYLE)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", edgecolor="none")

        # Set view angle
        ax.view_init(elev=30, azim=120)

        # Axis labels
        ax.set_xlabel("Hidden Size")
        ax.set_ylabel("Number of Hidden Layers")
        ax.set_zlabel("MSE")

        # Title
        ax.set_title("3D Surface View of Neural Network Performance")

        # Show color bar
        fig.colorbar(surf)

        # Gradient field (2D projection)
        fig2, ax2 = plt.subplots(figsize=(40, 30))
        contour = ax2.contourf(grid_x, grid_y, grid_z, cmap="viridis", alpha=0.7)
        quiver = ax2.quiver(grid_x, grid_y, grad_x, grad_y)
        ax2.set_xlabel("Hidden Size")
        ax2.set_ylabel("Number of Hidden Layers")
        ax2.set_title("Gradient of Neural Network MSE Surface")
        fig2.colorbar(contour)

        # Save figures
        fig.savefig("3D_Surface_View_of_Neural_Network_Performance.png")
        fig2.savefig("Gradient_of_Neural_Network_MSE_Surface.png")

        # Close figures
        plt.close()

    X_train = torch.tensor(data_loader.train_data, dtype=torch.float32)
    y_train = torch.tensor(data_loader.train_target, dtype=torch.float32)
    X_test = torch.tensor(data_loader.test_data, dtype=torch.float32)
    y_test = torch.tensor(data_loader.test_target, dtype=torch.float32)

    # Neural network regression
    nn_models = []
    nn_performance = []
    best_nn_model = None
    best_nn_mse = float("inf")  # Initialize to infinity
    for hidden_size in [16, 32, 64, 128]:
        for hidden_layers in [1, 2, 3, 4]:
            nn_model = NeuralNetworkRegression(
                input_size=1, hidden_size=hidden_size, num_hidden_layers=hidden_layers
            )
            nn_model.train_model(X_train, y_train, epochs=1000)
            performance = nn_model.evaluate_model(X_test, y_test, plot=True)
            nn_performance.append(performance)
            nn_models.append(nn_model)

            # Check if current model is the best
            if performance[2] < best_nn_mse:
                best_nn_mse = performance[2]
                best_nn_model = nn_model
                print(
                    f"Current best model: Hidden Size={hidden_size}, Layers={hidden_layers}, MSE={performance[2]:.4f}"
                )

    compare_nn_models(nn_models, X_test, y_test)

    plot_nn_performance(nn_performance)

    # ======================================================================================
    # Bayesian Regression ==================================================================
    # ======================================================================================

    def plot_bayesian_performance(performance_data: list[tuple[int, int, float]]):
        """
        Plot performance of Bayesian regression, including 3D surface and gradient field.

        Parameters:
        - performance_data (list[tuple[int, float, float]]): List of tuples containing (degree, tune, mse).
        """

        # Extract degrees, tunes, and MSE
        degrees = [record[0] for record in performance_data]
        tunes = [record[1] for record in performance_data]
        mse_values = [record[2] for record in performance_data]

        # Create grid data
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(degrees), max(degrees), 100),
            np.linspace(min(tunes), max(tunes), 100),
        )

        # Interpolate data
        grid_z = griddata(
            (degrees, tunes), mse_values, (grid_x, grid_y), method="linear"
        )

        # Compute gradients
        grad_x, grad_y = np.gradient(
            grid_z,
            np.linspace(min(degrees), max(degrees), 100),
            np.linspace(min(tunes), max(tunes), 100),
        )

        # Create 3D plot
        plt.style.use(PLOT_STYLE)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", edgecolor="none")

        # Set view angle
        ax.view_init(elev=45, azim=45)

        # Axis labels
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Tune")
        ax.set_zlabel("MSE")

        # Title
        ax.set_title("3D Surface View of Bayesian Regression Performance")

        # Show color bar
        fig.colorbar(surf)

        # Gradient field (2D projection)
        fig2, ax2 = plt.subplots(figsize=(40, 30))
        contour = ax2.contourf(grid_x, grid_y, grid_z, cmap="viridis", alpha=0.7)
        quiver = ax2.quiver(grid_x, grid_y, grad_x, grad_y)
        ax2.set_xlabel("Polynomial Degree")
        ax2.set_ylabel("Tune")
        ax2.set_title("Gradient of Bayesian Regression Performance")
        fig2.colorbar(contour)

        # Save figures
        fig.savefig("3D_Surface_View_of_Bayesian_Regression_Performance.png")
        fig2.savefig("Gradient_of_Bayesian_Regression_Performance.png")

        plt.close()

    # Bayesian regression
    bayesian_performance = []
    for degree in range(1, 6):
        for tune in [0, 100, 500, 1000, 2000]:
            bayesian_regression = BayesianPolynomialRegression(degree=degree, tune=tune)
            bayesian_regression.train(data_loader.train_data, data_loader.train_target)
            performance = bayesian_regression.evaluate(
                data_loader.test_data,
                data_loader.test_target,
                plot=True,
                X_train=data_loader.train_data,
            )
            bayesian_performance.append(performance)

    print(f"Bayesian regression models training and evaluation: {bayesian_performance}")
    plot_bayesian_performance(bayesian_performance)

    # ======================================================================================
    # Regression Test: Visualization for Training and Testing Data =========================
    # ======================================================================================

    def plot_regression_results(
        x_values: np.ndarray,
        y_values: np.ndarray,
        predictions: np.ndarray,
        mse: float,
        title: str,
        save_path: str,
        y_min: float,
        y_max: float,
        color: str = "green",
        model_label: str = "Predictions",
    ):
        """
        General function to plot regression results.

        Parameters:
        - x_values (np.ndarray): Input data (X values).
        - y_values (np.ndarray): Actual target values.
        - predictions (np.ndarray): Model predictions.
        - mse (float): Mean Squared Error.
        - title (str): Plot title.
        - save_path (str): Path to save the plot.
        - y_min (float): Minimum Y-axis value.
        - y_max (float): Maximum Y-axis value.
        - color (str): Color of the prediction line or points. Default is "green".
        - model_label (str): Legend label for model predictions. Default is "Predictions".
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(
            x_values,
            y_values,
            color="lightblue",
            alpha=0.7,
            label="Actual Values",
        )
        plt.plot(
            x_values,
            predictions,
            color=color,
            alpha=0.7,
            label=f"{model_label} (MSE={mse:.2f})",
        )
        plt.ylim(
            y_min - 0.1 * abs(y_min),
            y_max + 0.1 * abs(y_max),
        )
        plt.title(title)
        plt.xlabel("Input Data (X values)")
        plt.ylabel("Target / Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)

        plt.close()

    # 1. Load training and testing data
    train_loader = RegressionDataLoader()
    train_loader.load_data("regression_train.txt")

    test_loader = RegressionDataLoader()
    test_loader.load_data("regression_test.txt")

    # 2. Predict on training and testing data
    # Convert to Tensor
    x_train_tensor = torch.tensor(train_loader.data, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_loader.target, dtype=torch.float32)

    x_test_tensor = torch.tensor(test_loader.data, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_loader.target, dtype=torch.float32)

    # Linear regression predictions
    linear_train_predictions = best_linear_regression.model.predict(train_loader.data)
    linear_test_predictions = best_linear_regression.model.predict(test_loader.data)

    # Neural network predictions
    nn_train_predictions = (
        best_nn_model(x_train_tensor).squeeze().detach().cpu().numpy()
    )
    nn_test_predictions = best_nn_model(x_test_tensor).squeeze().detach().cpu().numpy()

    # 3. Compute MSE
    linear_train_mse = mean_squared_error(train_loader.target, linear_train_predictions)
    linear_test_mse = mean_squared_error(test_loader.target, linear_test_predictions)
    nn_train_mse = mean_squared_error(y_train_tensor, nn_train_predictions)
    nn_test_mse = mean_squared_error(y_test_tensor, nn_test_predictions)

    # Compute Y-axis ranges for linear regression model
    linear_y_min_train = min(train_loader.target.min(), linear_train_predictions.min())
    linear_y_max_train = max(train_loader.target.max(), linear_train_predictions.max())

    linear_y_min_test = min(test_loader.target.min(), linear_test_predictions.min())
    linear_y_max_test = max(test_loader.target.max(), linear_test_predictions.max())

    # Compute Y-axis ranges for neural network model
    nn_y_min_train = min(train_loader.target.min(), nn_train_predictions.min())
    nn_y_max_train = max(train_loader.target.max(), nn_train_predictions.max())

    nn_y_min_test = min(test_loader.target.min(), nn_test_predictions.min())
    nn_y_max_test = max(test_loader.target.max(), nn_test_predictions.max())

    # Linear Regression - Training Set
    plot_regression_results(
        x_values=train_loader.data,
        y_values=train_loader.target,
        predictions=linear_train_predictions,
        mse=linear_train_mse,
        title="Linear Regression - Training Set",
        save_path="linear_regression_training.png",
        y_min=linear_y_min_train,
        y_max=linear_y_max_train,
        color="cyan",
        model_label="Linear Regression Predictions",
    )

    # Linear Regression - Test Set
    plot_regression_results(
        x_values=test_loader.data,
        y_values=test_loader.target,
        predictions=linear_test_predictions,
        mse=linear_test_mse,
        title="Linear Regression - Test Set",
        save_path="linear_regression_test.png",
        y_min=linear_y_min_test,
        y_max=linear_y_max_test,
        color="cyan",
        model_label="Linear Regression Predictions",
    )

    # Neural Network Regression - Training Set
    plot_regression_results(
        x_values=train_loader.data,
        y_values=train_loader.target,
        predictions=nn_train_predictions,
        mse=nn_train_mse,
        title="Neural Network - Training Set",
        save_path="neural_network_training.png",
        y_min=nn_y_min_train,
        y_max=nn_y_max_train,
        color="orange",
        model_label="Neural Network Predictions",
    )

    # Neural Network Regression - Test Set
    plot_regression_results(
        x_values=test_loader.data,
        y_values=test_loader.target,
        predictions=nn_test_predictions,
        mse=nn_test_mse,
        title="Neural Network - Test Set",
        save_path="neural_network_test.png",
        y_min=nn_y_min_test,
        y_max=nn_y_max_test,
        color="orange",
        model_label="Neural Network Predictions",
    )

    # Plot combined training and testing data
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(8, 6))

    # Training data
    plt.scatter(
        train_loader.data,
        train_loader.target,
        color="lightblue",
        alpha=0.7,
        label="Training Data",
    )

    # Testing data
    plt.scatter(
        test_loader.data,
        test_loader.target,
        color="magenta",
        alpha=0.7,
        label="Testing Data",
    )

    plt.title("Full Dataset: Training and Testing Data")

    plt.xlabel("Input Data (X values)")
    plt.ylabel("Target Values")
    plt.legend()
    plt.grid(True)

    plt.savefig("full_dataset_training_testing_data.png")

    plt.close()
