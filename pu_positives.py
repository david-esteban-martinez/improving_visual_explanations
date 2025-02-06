import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import randint
from pytorch_lightning.callbacks import EarlyStopping, Callback


class CallbackEnd(Callback):
    """
    Custom callback for PyTorch Lightning that updates user centroids and
    resamples positive samples at the end of each training epoch.

    The DataFrame used in this process must contain the following columns:

    - `id_user` (int or str): Unique identifier for each user.
    - `id_img` (int or str): Unique identifier for each image.
    - `id_restaurant` (int or str): Unique identifier for a restaurant.
    - `take` (int: 0 or 1): Binary flag indicating whether an image is considered.

    These columns are required for the centroid computation and resampling functions.
    """

    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """
        Method executed at the end of each training epoch.

        Args:
            trainer: Instance of the PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        original_embeddings = trainer.train_dataloader.dataset.datamodule.image_embeddings
        dataframe = trainer.train_dataloader.dataset.dataframe
        dataframe = dataframe.drop_duplicates(keep='first').reset_index(drop=True)

        new_centroids = centroid_users(dataframe, original_embeddings.cpu().detach().numpy())
        trainer.train_dataloader.dataset.pu_dataset = resample_positives(dataframe, new_centroids)


def centroid_users(data: pd.DataFrame, vectors: np.ndarray, labels: list = None):
    """
    Computes user centroids based on their feature vectors and
    the 90th percentile distance between images and their centroid.

    Args:
        data (pd.DataFrame): DataFrame containing user and image data.
        vectors (np.ndarray): Matrix of image embeddings.

    Returns:
        dict: Dictionary with user centroids.
    """

    if labels is None:
        labels = ["id_user", "id_img"]

    data.rename(columns={labels[0]: "id_user", labels[1]: "id_img"},
              inplace=True)
    user_ids = data["id_user"]
    user_centroids = {}
    user_distances = {}
    img_ids = data["id_img"].to_numpy()[:, None]

    for user_id in user_ids.unique():
        user_indices = np.where(data["id_user"] == user_id)[0]
        vect = vectors[img_ids[user_indices]]

        # Compute the user centroid
        user_centroid = np.mean(vect, axis=0)
        user_centroids[user_id] = user_centroid

    return user_centroids


def resample_positives(dataframe: pd.DataFrame, centroids: dict, k: int = 3):
    """
    Generates new positive samples for each user based on computed centroids.

    Args:
        dataframe (pd.DataFrame): Original DataFrame.
        centroids (dict): Dictionary containing user centroids.
        k (int): Number of similar users to consider.
    Returns:
        pd.DataFrame: Updated DataFrame with new positive samples.
    """
    tqdm.pandas()
    new_positives = []

    for user in dataframe["id_user"].unique():
        # Randomly select 100 users to compute similarity
        random_users = np.random.randint(0, len(centroids), size=100)
        vect = [centroids[x] for x in random_users]
        user_centroid = centroids[user]

        # Adjust vector dimensionality
        vect = np.squeeze(np.array(vect), axis=1)
        norm = np.linalg.norm(vect, axis=1) * np.linalg.norm(user_centroid)
        distances = np.dot(vect, np.squeeze(user_centroid)) / norm

        # Get indices of the top-k closest users
        sorted_distances_index = sorted(range(len(distances.tolist())), key=lambda k: distances[k])
        top_k = random_users[sorted_distances_index[-k:]]

        for user_other in top_k:
            # Filter reviews from the selected user
            reviews_user = dataframe[dataframe["id_user"] == user_other]
            reviews_user = reviews_user[reviews_user["take"] == 1].drop_duplicates()
            random_review = reviews_user.sample()

            # Create a new positive sample
            new_sample = {
                "id_user": user,
                "id_restaurant": random_review["id_restaurant"].values[0],
                "id_img": random_review["id_img"].values[0],
                "take": 1
            }
            new_positives.append(new_sample)

    # Concatenate new samples with the original DataFrame
    new_dataframe = pd.concat([dataframe, pd.DataFrame(new_positives)])
    new_dataframe = new_dataframe.reset_index(drop=True)
    return new_dataframe
