import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


class CovtypeBunch(BaseModel):
    """
    Pydantic data model used to describe the data structure returned by fetch_covtype.
    """

    data: np.ndarray = Field(..., description="Sample feature matrix")
    target: np.ndarray = Field(..., description="Target classification labels")

    # Enable arbitrary_types_allowed configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataLoader:
    """
    Data loading class with type validation using Pydantic.
    """

    def __init__(self):
        self.data: Optional[CovtypeBunch] = None  # Complete data
        self.processed_data: Optional[np.ndarray] = None  # Processed data
        self.sample_data: Optional[np.ndarray] = None  # Sampled data
        self.sample_labels: Optional[np.ndarray] = None  # Sampled labels

    def load_data(self) -> "DataLoader":
        """
        Load the Covertype dataset from sklearn and validate with Pydantic model.
        """
        raw_data = fetch_covtype()  # Bunch object returned by sklearn
        # print(raw_data.keys())  # dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
        self.data = CovtypeBunch(data=raw_data.data, target=raw_data.target)
        print(f"Dataset loaded and validated. Data dimensions: {self.data.data.shape}")

        return self

    def clean_data(self) -> "DataLoader":
        """
        Data cleaning: remove NaN and Inf values.
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Please call load_data() first.")

        if np.isnan(self.data.data).any() or np.isinf(self.data.data).any():
            print("Anomalous values found. Removing...")
            mask = np.all(np.isfinite(self.data.data), axis=1)
            self.data.data = self.data.data[mask]
            print(
                f"Anomalous values removed. Remaining data points: {self.data.data.shape[0]}"
            )

        return self

    def scale_data(self) -> "DataLoader":
        """
        Data standardization: convert feature values to distribution with mean 0 and standard deviation 1.
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Please call load_data() first.")

        scaler = StandardScaler()
        self.processed_data = scaler.fit_transform(self.data.data)
        print("Data standardization complete.")

        return self

    def apply_pca(self, variance_ratio: float = 0.90) -> "DataLoader":
        """
        Apply PCA to reduce data dimensionality, dynamically selecting the number of dimensions that satisfy the target explained variance ratio.

        Parameters:
        - variance_ratio (float): Target explained variance ratio, default value is 0.90.
        """
        if self.processed_data is None:
            raise ValueError(
                "Data not standardized yet. Please call scale_data() first."
            )

        # Initial PCA
        pca = PCA()
        pca.fit(self.processed_data)

        # Calculate cumulative explained variance ratio
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find the minimum number of dimensions that satisfy the target explained variance ratio
        n_components = np.argmax(cumulative_variance >= variance_ratio) + 1

        # Re-apply PCA with dynamically calculated number of dimensions
        pca = PCA(n_components=n_components)
        self.processed_data = pca.fit_transform(self.processed_data)

        # Print results
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(
            f"PCA dimensionality reduction complete, reduced to {n_components} dimensions, explained variance ratio: {explained_variance:.2%}"
        )

        return self

    def random_sample(self, n_samples: int = 10000) -> "DataLoader":
        """
        Randomly select n_samples data points.

        Parameters:
        - n_samples (int): Number of samples to draw, default value is 10000.
        """
        if self.processed_data is None:
            raise ValueError(
                "Data not dimensionally reduced yet. Please call scale_data() method."
            )

        # Extract data from the validated Pydantic data model
        indices: np.ndarray = np.random.choice(
            self.processed_data.shape[0],
            size=n_samples,
            replace=False,
        )
        self.sample_data = self.processed_data[indices]
        self.sample_labels = self.data.target[indices]
        print(f"Randomly selected {n_samples} data points.")

        return self


class KMeansClustering:
    """
    Task 2: Use K-means clustering.
    Responsible for performing K-means clustering on data.
    """

    def __init__(self, n_clusters: int = 7) -> None:
        """
        Initialize KMeansClustering instance.

        Parameters:
        - n_clusters (int): Number of clusters, default value is 7.
        """
        self.n_clusters: int = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None  # Save cluster labels

    def cluster(self, data: np.ndarray) -> None:
        """
        Perform K-means clustering on the data and print evaluation metrics.
        """
        # Use K-means++
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            random_state=42,
            n_init=50,
            algorithm="elkan",  # Use "elkan" to speed up computation
        )
        self.kmeans.fit(data)
        print("K-means clustering complete.")

        # Save clustering results
        self.cluster_labels = self.kmeans.labels_ + 1  # Cluster labels for each sample

        # Print clustering evaluation metrics
        inertia = self.kmeans.inertia_
        silhouette = silhouette_score(data, self.kmeans.labels_)
        print(f"K-means Clustering: Inertia: {inertia:.2f}")
        print(f"K-means Clustering: Silhouette Score: {silhouette:.2f}")


class GMMClustering:
    """
    Task 3: Use Gaussian Mixture Model (GMM) clustering.
    Responsible for performing GMM clustering on data.
    """

    def __init__(self, n_components: int = 7) -> None:
        """
        Initialize GMMClustering instance.

        Parameters:
        - n_components (int): Number of components in GMM, default value is 7.
        """
        self.n_components: int = n_components
        self.gmm: Optional[GaussianMixture] = None
        self.cluster_labels: Optional[np.ndarray] = None  # Save cluster labels

    def cluster(self, data: np.ndarray) -> None:
        """
        Perform GMM clustering on the data and save results.

        Parameters:
        - data (np.ndarray): Input data, shape (n_samples, n_features).
        """
        # Use K-means initialization
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",  # Use the same covariance matrix
            random_state=42,
            n_init=5,
            init_params="kmeans",
            reg_covar=1e-4,
        )
        self.gmm.fit(data)

        # Save clustering results
        self.cluster_labels = (
            self.gmm.predict(data) + 1
        )  # Cluster labels for each sample

        # Print results
        print("GMM clustering complete.")
        log_likelihood = self.gmm.score(data) * len(data)
        silhouette = silhouette_score(data, self.cluster_labels)
        print(f"GMM Clustering: Log-likelihood: {log_likelihood:.2f}")
        print(f"GMM Clustering: Silhouette Score: {silhouette:.2f}")


class RandomBaseline:
    """
    Task 4: Random baseline.
    Randomly assign a cluster label to each data point.
    """

    def __init__(self, n_clusters: int = 7) -> None:
        """
        Initialize RandomBaseline instance.

        Parameters:
        - n_clusters (int): Number of clusters, default value is 7.
        """
        self.n_clusters: int = n_clusters
        self.labels: Optional[np.ndarray] = None

    def assign_random_clusters(self, data: np.ndarray) -> None:
        """
        Randomly assign cluster labels to each data point.

        Parameters:
        - data (np.ndarray): Input data, shape (n_samples, n_features).
        """
        self.labels = np.random.choice(
            range(1, self.n_clusters + 1),
            size=data.shape[0],
        )
        print("Random cluster label assignment complete.")


class ErrorEvaluator:
    """
    Task 5: Error evaluation.
    Class for evaluating clustering algorithm error rates.
    """

    def __init__(self, target_labels: np.ndarray):
        """
        Initialize ErrorEvaluator.

        Parameters:
        - target_labels (np.ndarray): True class labels of the data points, shape (n_samples,).
        """
        self.target_labels = target_labels

    def count_pairwise_errors(
        self,
        cluster_labels: np.ndarray,
    ) -> int:
        """
        Calculate errors as per task requirement: For each pair of data points with the same true label,
        if they are assigned to different clusters, count as one error.

        Parameters:
        - cluster_labels (np.ndarray): Predicted labels from the clustering algorithm.

        Returns:
        - int: Number of erroneous data point pairs.
        """
        # Initialize error count
        error_count = 0

        # Iterate over each unique true label
        unique_targets = np.unique(self.target_labels)
        for target in tqdm(unique_targets, desc="Calculating errors"):
            # Get indices of data points with the true label 'target'
            indices = np.where(self.target_labels == target)[0]

            # Skip labels with only one point
            if len(indices) < 2:
                continue

            # Extract corresponding cluster labels
            sub_cluster_labels = cluster_labels[indices]

            # Create a boolean matrix indicating whether pairs are assigned to different clusters
            pairwise_diff = sub_cluster_labels[:, None] != sub_cluster_labels

            # Count errors: Number of True values in the boolean matrix
            error_count += np.sum(pairwise_diff)

        # Return the number of errors
        return error_count // 2  # Each pair counted twice, divide by 2

    def count_total_pairs(self) -> int:
        """
        Calculate the total number of same-class data point pairs.

        Returns:
        - int: Total number of same-class data point pairs.
        """
        total_pairs = 0

        # Iterate over each unique true label
        unique_targets = np.unique(self.target_labels)
        for target in tqdm(unique_targets, desc="Calculating total same-class pairs"):
            # Get indices of data points with the true label 'target'
            indices = np.where(self.target_labels == target)[0]

            # Skip labels with only one point
            if len(indices) < 2:
                continue

            # Count total pairs
            total_pairs += len(indices) * (len(indices) - 1) // 2

        return total_pairs


if __name__ == "__main__":
    import os

    os.environ["LOKY_MAX_CPU_COUNT"] = "4"

    # Set random seed
    np.random.seed(42)

    # Initialize task classes
    kmeans_cluster = KMeansClustering()
    gmm_cluster = GMMClustering()
    random_baseline = RandomBaseline()

    # Complete tasks in order
    # Task 1: Load data and random sampling
    # Instantiate DataLoader
    loader = DataLoader()
    # Sequentially call methods: load data, clean, standardize, random sample
    loader.load_data().clean_data().scale_data().random_sample()

    # Task 2: Use K-means clustering
    kmeans_cluster.cluster(loader.sample_data)

    # Task 3: Use GMM clustering
    gmm_cluster.cluster(loader.sample_data)

    # Task 4: Random baseline
    random_baseline.assign_random_clusters(loader.sample_data)

    # Task 5: Evaluate clustering results
    evaluator = ErrorEvaluator(loader.sample_labels)
    kmeans_errors = evaluator.count_pairwise_errors(kmeans_cluster.cluster_labels)
    gmm_errors = evaluator.count_pairwise_errors(gmm_cluster.cluster_labels)
    baseline_errors = evaluator.count_pairwise_errors(random_baseline.labels)
    total_pairs = evaluator.count_total_pairs()

    # Print error rates
    print(f"K-means errors: {kmeans_errors}")
    print(f"GMM errors: {gmm_errors}")
    print(f"Random baseline errors: {baseline_errors}")
    print(f"Total same-class data point pairs: {total_pairs}")
