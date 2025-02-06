import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import timm
from PIL import Image
import torch
from timm.data import resolve_data_config, create_transform
from torchvision import transforms

# Dictionary containing predefined image transformations
TRANSFORMATIONS = {
    "random_perspective": transforms.RandomPerspective(p=1),
    "gaussian_blur": transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    "random_erase": transforms.RandomErasing(p=1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
}


def X_transform(img, apply_all=False):
    """
    Applies a random transformation to the given image.

    Args:
        img (PIL.Image): The image to transform.
        apply_all (bool): If True, applies all transformations and returns a list.

    Returns:
        Transformed image(s) as PIL.Image.
    """
    if apply_all:
        return [t(img) for t in TRANSFORMATIONS.values()]
    else:
        # Apply one randomly selected transformation
        random_key = random.choice(list(TRANSFORMATIONS.keys()))
        if random_key == "random_erase":
            img = transforms.ToTensor()(img)
            img = TRANSFORMATIONS[random_key](img)
            img = transforms.ToPILImage()(img)
        else:
            img = TRANSFORMATIONS[random_key](img)
        return img


def transform_image(filename, directory, num_images, transform, apply_all=False, no_aug=False):
    """
    Loads and transforms an image from a given directory.

    Args:
        filename (str): Name of the image file.
        directory (str): Path to the directory containing images.
        num_images (int): Counter for generated image IDs.
        transform (callable): Function to apply transformations.
        apply_all (bool): If True, applies all transformations.
        no_aug (bool): If True, avoids transformations and returns the original image.

    Returns:
        pd.DataFrame: A DataFrame containing metadata for the transformed images.
        list: List of transformed image tensors.
    """
    filepath = os.path.join(directory, filename)
    img = Image.open(filepath).convert('RGB')
    samples = []
    user, restaurant = filename.split("_")[:2]

    if not no_aug:
        transformed_images = X_transform(img, apply_all=apply_all)
        if not isinstance(transformed_images, list):
            transformed_images = [transformed_images]
        for _ in transformed_images:
            new_sample = {
                "id_user": int(user),
                "id_restaurant": int(restaurant),
                "id_img": num_images,
                "take": 1
            }
            num_images += 1
            samples.append(new_sample)
    else:
        transformed_images = [img]
        new_sample = {
            "id_user": int(user),
            "id_restaurant": int(restaurant),
            "id_img": num_images,
            "take": 1
        }
        num_images += 1
        samples.append(new_sample)

    transformed_tensors = [transform(t_img) for t_img in transformed_images]
    return pd.DataFrame(samples), transformed_tensors


def process_images(file_list, directory, num_images, transform, apply_all=False, no_aug=False):
    """
    Processes a batch of images, applying transformations and collecting results.

    Args:
        file_list (list): List of filenames.
        directory (str): Path to the image directory.
        num_images (int): Counter for image IDs.
        transform (callable): Transformation function.
        apply_all (bool): If True, applies all transformations.
        no_aug (bool): If True, avoids transformations.

    Returns:
        list: Processed images with metadata.
        int: Number of failed processing attempts.
    """
    results = []
    num_failures = 0
    for filename in file_list:
        try:
            synthetic_df, tensors = transform_image(filename, directory, num_images, transform, apply_all, no_aug)
            num_images += len(tensors)
            results.append((synthetic_df, tensors))
        except Exception as e:
            num_failures += 1
            print("FAILED: " + str(e))
            continue
    return results, num_failures


def include_negatives(data_add, data_search):
    """
    Generates negative samples by selecting images from different users and restaurants.

    Args:
        data_add (pd.DataFrame): DataFrame containing positive samples.
        data_search (pd.DataFrame): DataFrame containing all available images.

    Returns:
        pd.DataFrame: DataFrame containing both positive and negative samples.
    """
    tqdm.pandas()
    dictionary_dataframe = []
    data_add = data_add[data_add["take"] == 1]
    for _ in range(1):
        neg_samples = (
            data_add.groupby("id_user", group_keys=False)
            .progress_apply(lambda x: getSamplesDifferentRestaurant(x, data_search))
            .reset_index(drop=True)
        )
        dictionary_dataframe.extend(neg_samples.to_dict(orient="records"))
        dictionary_dataframe.extend(data_add.to_dict(orient="records"))

    # Generate negative samples within the same restaurant
    data = (
        data_add.groupby("id_restaurant")
        .filter(lambda x: x["id_user"].nunique() > 1)
        .reset_index(drop=True)
    )
    for _ in range(1):
        same_res_bpr_samples = (
            data.groupby("id_restaurant", group_keys=False)
            .progress_apply(lambda x: getSamplesSameRestaurant(x))
            .reset_index(drop=True)
        )
        dictionary_dataframe.extend(same_res_bpr_samples.to_dict(orient="records"))

    dataframe = pd.DataFrame.from_dict(dictionary_dataframe)
    print("Positive samples: ", dataframe[dataframe["take"] == 1].shape[0])
    print("Negative samples: ", dataframe[dataframe["take"] == 0].shape[0])
    return dataframe


def getSamplesSameRestaurant(data_rest):
    """
    Selects negative samples within the same restaurant by reassigning images.

    Args:
        data_rest (pd.DataFrame): Data from a single restaurant.

    Returns:
        pd.DataFrame: Updated DataFrame with negative samples.
    """
    user_ids = data_rest["id_user"].to_numpy()[:, None]
    img_ids = data_rest["id_img"].to_numpy()[:, None]
    new_negatives = np.random.randint(len(data_rest), size=len(data_rest))
    counter = 0
    while True:
        invalid_samples = (user_ids[new_negatives] == user_ids).flatten()
        num_invalid_samples = np.sum(invalid_samples)
        if num_invalid_samples == 0 or counter > 100:
            break
        new_negatives[invalid_samples] = np.random.randint(data_rest.shape[0], size=num_invalid_samples)
        counter += 1
    data_rest["id_img"] = img_ids[new_negatives]
    data_rest["take"] = 0
    return data_rest


def getSamplesDifferentRestaurant(data_user, data):
    """
    Selects negative samples from different restaurants to create a balanced dataset.

    Args:
        data_user (pd.DataFrame): DataFrame of user images.
        data (pd.DataFrame): Full dataset.

    Returns:
        pd.DataFrame: Updated DataFrame with negative samples.
    """
    user_ids = data["id_user"].to_numpy()[:, None]
    img_ids = data["id_img"].to_numpy()[:, None]
    rest_ids = data["id_restaurant"].to_numpy()[:, None]

    id_user = data_user["id_user"].values[0]
    rest_user = data_user["id_restaurant"].to_numpy()[:, None]

    new_negatives = np.random.randint(data.shape[0], size=data_user.shape[0])
    counter = 0
    while True:
        invalid_samples = (
                (user_ids[new_negatives] == id_user).flatten() |
                (rest_ids[new_negatives] == rest_user).flatten()
        )
        num_invalid_samples = np.sum(invalid_samples)
        counter += 1
        if num_invalid_samples == 0 or counter == 100:
            break
        new_negatives[invalid_samples] = np.random.randint(data.shape[0], size=num_invalid_samples)

    data_user["id_img"] = img_ids[new_negatives]
    data_user["take"] = 0
    data_user["id_restaurant"] = rest_ids[new_negatives]
    return data_user


def main(data_dir, vector_dir, image_dir, output_dir, output_name="TRAIN_IMG",
         embedding_model=None, no_aug=False, batch_size=32, apply_all=False, labels=None):
    """
    Main function to process images, apply transformations, and balance the dataset.

    Args:
        data_dir (str): Path to input dataset.
        vector_dir (str): Path to vectorized image embeddings.
        image_dir (str): Path to directory containing images.
        output_dir (str): Path to store processed output.
        output_name (str): Output filename.
        embedding_model (torch model, optional): Pretrained model for embedding extraction.
        no_aug (bool): If True, skips augmentation.
        batch_size (int): Batch size for processing.
        apply_all (bool): If True, applies all transformations, if False, one transformation is applied randomly to each image.
        labels (list, optional): Column labels for the dataset.
    """
    # Load image metadata and embeddings
    images = pickle.load(open(data_dir, "rb"))
    image_vec = pickle.load(open(vector_dir, "rb"))
    if labels is None:
        labels = ["id_user", "id_img", "id_restaurant", "take"]

    # Standardize column names
    images.rename(columns={labels[0]: "id_user", labels[1]: "id_img", labels[2]: "id_restaurant", labels[3]: "take"},
                  inplace=True)

    # Load or initialize the embedding model
    if embedding_model:
        model = embedding_model
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    else:
        model = timm.create_model('inception_resnet_v2', pretrained=True).to("cuda")
        model.eval()
        model.classif = torch.nn.Identity()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

    # Get the next available image ID
    num_images = images["id_img"].max() + 1
    file_list = [file for file in os.listdir(image_dir)]
    batch_list = []

    # Process images in batches
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i + batch_size]
        results, num_failures = process_images(batch_files, image_dir, num_images, transform, apply_all, no_aug)
        num_images += batch_size * (len(TRANSFORMATIONS) if apply_all else 1)
        num_images -= num_failures

        batch_dfs, batch_tensors = [], []
        for synthetic_df, tensors in results:
            batch_dfs.append(synthetic_df)
            batch_tensors.extend(tensors)

        # Convert batch of transformed images to tensor and extract embeddings
        batch_tensors = torch.stack(batch_tensors).to('cuda')
        with torch.amp.autocast("cuda"), torch.no_grad():
            batch_out = model(batch_tensors)

        # Append new image data to dataset
        images = pd.concat([images] + batch_dfs, ignore_index=True)
        batch_list.append(batch_out)

    # Convert extracted embeddings to numpy array
    batch_array = np.vstack([t.cpu().numpy() for t in batch_list])
    image_data = np.vstack([images, batch_array])
    os.makedirs(output_dir, exist_ok=True)

    # Save updated dataset and embeddings
    images.to_pickle(f"{output_dir}/{output_name}")
    with open(f"{output_dir}/IMG_VEC", 'wb') as f:
        pickle.dump(image_data, f)
    print("Processing complete. Data saved successfully.")


