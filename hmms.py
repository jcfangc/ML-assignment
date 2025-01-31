import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

PLOT_STYLE = "dark_background"


def plot_transition_matrix_heatmap(
    transmat, save_name: str, title="Transition Matrix Heatmap"
):
    """
    Visualize the heatmap of the transition probability matrix.

    Parameters:
    - transmat: Transition probability matrix, shape (n_states, n_states).
    - title: Title of the image.
    """
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        xticklabels=range(transmat.shape[0]),
        yticklabels=range(transmat.shape[0]),
    )
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.savefig(f"Transition_Matrix_{save_name}.png")


def plot_emission_matrix_heatmap(
    emissionprob, save_name: str, title="Emission Probability Heatmap"
):
    """
    Visualize the heatmap of the emission probability matrix.

    Parameters:
    - emissionprob: Emission probability matrix, shape (n_states, n_observations).
    - title: Title of the image.
    """
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        emissionprob,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        xticklabels=range(emissionprob.shape[1]),
        yticklabels=range(emissionprob.shape[0]),
    )
    plt.title(title)
    plt.xlabel("Observation")
    plt.ylabel("Hidden State")
    plt.savefig(f"Emission_Matrix_{save_name}.png")


def plot_start_probabilities_bar(
    startprob, save_name: str, title="Start Probabilities"
):
    """
    Visualize the bar chart of start probabilities.

    Parameters:
    - startprob: Array of start probabilities, shape (n_states,).
    - title: Title of the image.
    """
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(startprob)), startprob, color="skyblue", alpha=0.8)
    plt.title(title)
    plt.xlabel("Hidden State")
    plt.ylabel("Probability")
    plt.xticks(range(len(startprob)))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"Start_Probabilities_{save_name}.png")


def load_rewards(file_path: str) -> np.ndarray:
    """
    Read the reward data file and format it for HMM input.
    The file contains a one-dimensional sequence of integers, each value being 0, 1, or 2.

    Parameters:
    - file_path: Path to the reward data file.

    Returns:
    - np.ndarray: Array of reward sequences.
    """
    try:
        rewards = np.loadtxt(file_path, dtype=int)
        print(f"Loaded rewards sequence of length: {len(rewards)}")
        return rewards
    except Exception as e:
        print(f"Error reading file: {e}")
        raise


# Code Task 14: Training without true transition probabilities
def train_hmm_without_true_transitions(
    rewards: np.ndarray, n_states: int = 9, n_iter: int = 100
):
    """
    Train an HMM on reward data using the EM algorithm, estimating all parameters.

    Parameters:
    - rewards: Reward data, shape (n_samples,).
    - n_states: Number of hidden states in the HMM, default is 9 (number of grid points).
    - n_iter: Maximum number of iterations for the EM algorithm.

    Returns:
    - model: Trained HMM model.
    """
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iter, random_state=42)
    model.fit(rewards.reshape(-1, 1))  # HMM input must be a 2D array
    print("Training complete. Estimated parameters:")
    print(f"Start probabilities: {model.startprob_}")
    print(f"Transition matrix: {model.transmat_}")
    print(f"Emission probabilities: {model.emissionprob_}")
    return model


# Code Task 15: Training with true transition probabilities
def train_hmm_with_true_transitions(
    rewards: np.ndarray, true_transmat: np.ndarray, n_states: int = 9, n_iter: int = 100
):
    """
    Train an HMM on reward data using the EM algorithm, with the true transition probabilities fixed.

    Parameters:
    - rewards: Reward data, shape (n_samples,).
    - true_transmat: The true transition probability matrix.
    - n_states: Number of hidden states in the HMM, default is 9.
    - n_iter: Maximum number of iterations for the EM algorithm.

    Returns:
    - model: Trained HMM model.
    """
    model = hmm.MultinomialHMM(
        n_components=n_states,
        n_iter=n_iter,
        params="se",  # Optimize startprob and emissionprob
        init_params="se",  # Initialize startprob and emissionprob
        random_state=42,
    )
    model.transmat_ = true_transmat  # Set the true transition matrix
    model.fit(rewards.reshape(-1, 1))  # HMM input must be a 2D array
    print("Training complete with fixed transition matrix. Estimated parameters:")
    print(f"Start probabilities: {model.startprob_}")
    print(f"Emission probabilities: {model.emissionprob_}")
    return model


def generate_true_transition_matrix(grid_size: int = 3) -> np.ndarray:
    """
    Generate the true transition probability matrix for a grid world.

    Parameters:
    - grid_size: Size of the grid (number of rows/columns), default is 3x3.

    Returns:
    - np.ndarray: Transition probability matrix, shape (n_states, n_states).
    """
    n_states = grid_size * grid_size
    transmat = np.zeros((n_states, n_states))

    for x in range(grid_size):
        for y in range(grid_size):
            current_state = x * grid_size + y
            neighbors = []

            # Check neighboring states
            if x > 0:
                neighbors.append((x - 1) * grid_size + y)  # Up
            if x < grid_size - 1:
                neighbors.append((x + 1) * grid_size + y)  # Down
            if y > 0:
                neighbors.append(x * grid_size + y - 1)  # Left
            if y < grid_size - 1:
                neighbors.append(x * grid_size + y + 1)  # Right

            # Assign equal transition probabilities
            for neighbor in neighbors:
                transmat[current_state, neighbor] = 1 / len(neighbors)

    return transmat


if __name__ == "__main__":
    # Load reward data
    rewards = load_rewards("rewards.txt")

    # Generate the true transition probability matrix
    true_transmat = generate_true_transition_matrix(grid_size=3)
    print("True Transition Matrix:")
    print(true_transmat)

    # Code Task 14: Training without true transition probabilities
    print("\nCode Task 14: Training without true transition probabilities")
    model_without_transitions = train_hmm_without_true_transitions(rewards)

    # Code Task 15: Training with true transition probabilities
    print("\nCode Task 15: Training with true transition probabilities")
    model_with_transitions = train_hmm_with_true_transitions(rewards, true_transmat)

    SAVE_NAME_FOR_WITHOUT_TRANSITIONS = "without_true_transitions"
    SAVE_NAME_FOR_WITH_TRANSITIONS = "with_true_transitions"

    # Visualize training results
    print("\nVisualizing estimated parameters for HMM without true transitions:")
    plot_transition_matrix_heatmap(
        model_without_transitions.transmat_,
        SAVE_NAME_FOR_WITHOUT_TRANSITIONS,
        "Estimated Transition Matrix (Without True Transitions)",
    )
    plot_emission_matrix_heatmap(
        model_without_transitions.emissionprob_,
        SAVE_NAME_FOR_WITHOUT_TRANSITIONS,
        "Estimated Emission Matrix (Without True Transitions)",
    )
    plot_start_probabilities_bar(
        model_without_transitions.startprob_,
        SAVE_NAME_FOR_WITHOUT_TRANSITIONS,
        "Estimated Start Probabilities (Without True Transitions)",
    )

    print("\nVisualizing estimated parameters for HMM with true transitions:")
    plot_transition_matrix_heatmap(
        model_with_transitions.transmat_,
        SAVE_NAME_FOR_WITH_TRANSITIONS,
        "Transition Matrix (With True Transitions)",
    )
    plot_emission_matrix_heatmap(
        model_with_transitions.emissionprob_,
        SAVE_NAME_FOR_WITH_TRANSITIONS,
        "Estimated Emission Matrix (With True Transitions)",
    )
    plot_start_probabilities_bar(
        model_with_transitions.startprob_,
        SAVE_NAME_FOR_WITH_TRANSITIONS,
        "Estimated Start Probabilities (With True Transitions)",
    )
