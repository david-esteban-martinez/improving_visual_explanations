import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def get_pickle(path, name):
    """
    Loads a pickle file from a specified directory.

    Args:
        path (str): Path to the directory containing the pickle file.
        name (str): Name of the pickle file.

    Returns:
        data: Loaded pickle data.
    """
    with open(path + name, "rb") as handle:
        data = pickle.load(handle)
    return data


def CR_VSM(data, vectores, centroid, factor):
    """
    Generates negative samples by computing user centroids and selecting negatives
    from different and same restaurants.

    Args:
        data (pd.DataFrame): Data containing user interactions.
        vectores (np.ndarray): Image embeddings.
        centroid (int): Centroid threshold percentile.
        factor (float): Scaling factor for distance thresholds.

    Returns:
        pd.DataFrame: Processed data with balanced positive and negative samples.
    """
    dictionary_dataframe = []
    tqdm.pandas()

    print("Calculating centroid for the users ...")
    centroids, distances = centroid_users(data, vectores, centroid, factor)

    print("Getting negatives from different restaurants...")
    for _ in range(10):
        neg_samples = (
            data.groupby("id_user", group_keys=False)
            .progress_apply(lambda x: getSamplesDifferentRestaurantCentroid(x, data, centroids, distances, vectores))
            .reset_index(drop=True)
        )
        dictionary_dataframe.extend(neg_samples.to_dict(orient="records"))
        dictionary_dataframe.extend(data.to_dict(orient="records"))

    print("Getting negatives from same restaurants...")

    # Filter restaurants that have images from multiple users
    data = data.groupby("id_restaurant").filter(lambda x: x["id_user"].nunique() > 1).reset_index(drop=True)

    for _ in range(10):
        same_res_bpr_samples = (
            data.groupby("id_restaurant", group_keys=False)
            .progress_apply(lambda x: getSamplesSameRestaurantCentroid(x, centroids, distances, vectores))
            .reset_index(drop=True)
        )
        dictionary_dataframe.extend(same_res_bpr_samples.to_dict(orient="records"))
        dictionary_dataframe.extend(data.to_dict(orient="records"))

    print("Creating dataframe...")
    dataframe = pd.DataFrame.from_dict(dictionary_dataframe)

    # Ensure each user has the same number of positive and negative samples
    assert np.all(
        dataframe[dataframe["take"] == 1].groupby("id_user").size() ==
        dataframe[dataframe["take"] == 0].groupby("id_user").size()
    )

    # Ensure each user has at least 10 positive samples
    assert np.all(dataframe[dataframe["take"] == 1].groupby("id_user").size() >= 10)

    return dataframe


def centroid_users(data, vectores, centroid, factor):
    """
    Computes user centroids and the 90th percentile distances.

    Args:
        data (pd.DataFrame): User interaction data.
        vectores (np.ndarray): Image embeddings.
        centroid (int): Centroid percentile threshold.
        factor (float): Scaling factor for distance threshold.

    Returns:
        np.ndarray: Array of user centroids.
        np.ndarray: Array of user distances.
    """
    user_ids = data["id_user"].unique()
    user_centroids = np.zeros((len(user_ids), vectores.shape[1]))
    user_distances = np.zeros(len(user_ids))
    img_ids = data["id_img"].to_numpy()[:, None]

    for idx, user_id in enumerate(user_ids):
        user_indices = np.where(data["id_user"] == user_id)[0]
        vect = vectores[img_ids[user_indices]]
        user_centroid = np.mean(vect, axis=0)
        user_centroids[idx] = user_centroid

        # Compute cosine distances
        norm = np.linalg.norm(vect, axis=1) * np.linalg.norm(user_centroid)
        distances = np.dot(vect, user_centroid) / norm

        # Compute percentile threshold
        p90 = np.percentile(distances, centroid)
        user_distances[idx] = p90 / factor

    return user_centroids, user_distances


def getSamplesDifferentRestaurantCentroid(data_user, data, centroid_ids, p90, vectores):
    """
    Generates negative samples from different restaurants while ensuring they are
    dissimilar to the user's centroid.
    """
    user_ids = data["id_user"].to_numpy()[:, None]
    img_ids = data["id_img"].to_numpy()[:, None]
    rest_ids = data["id_restaurant"].to_numpy()[:, None]

    id_user = data_user["id_user"].values[0]
    rest_user = data_user["id_restaurant"].to_numpy()[:, None]

    new_negatives = np.random.randint(data.shape[0], size=data_user.shape[0])
    centr = centroid_ids[id_user]
    counter = 0

    while True:
        # Identify invalid samples that belong to the same user or restaurant
        invalid_samples = (
                (user_ids[new_negatives] == id_user).flatten() |
                (rest_ids[new_negatives] == rest_user).flatten() |
                (np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1) / (
                        np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1), axis=1) * np.linalg.norm(
                    centr)
                ) > p90[id_user])
        )
        num_invalid_samples = np.sum(invalid_samples)
        counter += 1
        if num_invalid_samples == 0 or counter == 100:
            break
        # Reassign invalid samples randomly
        new_negatives[invalid_samples] = np.random.randint(data.shape[0], size=num_invalid_samples)

    # Assign new negative samples
    data_user["id_img"] = img_ids[new_negatives]
    data_user["take"] = 0
    data_user["id_restaurant"] = rest_ids[new_negatives]

    return data_user


def getSamplesSameRestaurantCentroid(data_rest, centroid_ids, p90, vectores):
    """
    Generates negative samples within the same restaurant, ensuring they differ
    based on the user's centroid similarity threshold.
    """
    user_ids = data_rest["id_user"].to_numpy()[:, None]
    img_ids = data_rest["id_img"].to_numpy()[:, None]

    new_negatives = np.random.randint(len(data_rest), size=len(data_rest))
    counter = 0
    centr = np.squeeze(centroid_ids[user_ids.flatten()], axis=(1))

    while True:
        # Identify invalid samples that belong to the same user or exceed similarity threshold
        invalid_samples = (
                (user_ids[new_negatives] == user_ids).flatten() |
                (np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1) / (
                        np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1), axis=1) * np.linalg.norm(
                    centr)
                ) > p90[user_ids.flatten()])
        )
        num_invalid_samples = np.sum(invalid_samples)
        if num_invalid_samples == 0 or counter > 100:
            break
        # Reassign invalid samples randomly
        new_negatives[invalid_samples] = np.random.randint(data_rest.shape[0], size=num_invalid_samples)
        counter += 1

    # Assign new negative samples
    data_rest["id_img"] = img_ids[new_negatives]
    data_rest["take"] = 0

    return data_rest


def main(data_file, vector_file, outdir_name, centroid=90, factor=1., labels=None):
    """
    Main function to generate negative samples and balance dataset.

    Args:
        data_file (str): Path to user interaction data.
        vector_file (str): Path to image embedding file.
        outdir_name (str): Output directory.
        centroid (int): Percentile threshold for centroids.
        factor (float): Distance scaling factor.
        labels (list, optional): Column labels.
    """
    print("Loading data...")
    datos = pickle.load(open(data_file, "rb"))
    vectores = np.array(pickle.load(open(vector_file, "rb")))
    df = pd.DataFrame(datos)

    # Standardize column names

    if labels is None:
        labels = ["id_user", "id_img", "id_restaurant", "take"]

    df.rename(columns={labels[0]: "id_user", labels[1]: "id_img", labels[2]: "id_restaurant", labels[3]: "take"},
              inplace=True)

    # Filter out entries where take is 0
    df = df[df["take"] != 0]
    df = df.drop_duplicates(subset=["id_img"], keep='first')

    print("Generating negative samples...")
    new_dataframe = CR_VSM(df, vectores, centroid, factor).reset_index(drop=True)

    os.makedirs(outdir_name, exist_ok=True)

    if labels is not None:
        new_dataframe.rename(
            columns={"id_user": labels[0], "id_img": labels[1], "id_restaurant": labels[2], "take": labels[3]},
            inplace=True)

    print("Saving processed dataset...")
    new_dataframe.to_pickle(os.path.join(outdir_name, "TRAIN_IMG"))
    print("Process completed successfully.")
