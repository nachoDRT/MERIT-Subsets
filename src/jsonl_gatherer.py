from utils import *
import requests
import argparse


TRAINING_SAMPLES = 150
FOLDER_BASE = "openai-vfinetune"
FOLDER = "openai-vfinetune"
SUBFOLDER = "data"


def set_folder(subset: str):

    global FOLDER
    FOLDER = f"{FOLDER_BASE}-{subset}"


def get_repo_config():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)

    owner = secrets["owner"]
    repo = secrets["repo"]
    token = secrets["token"]
    branch = "main"

    return owner, token, repo, branch


def compose_url(owner: str, repo: str, branch: str, image_name: str):

    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{FOLDER}/{SUBFOLDER}/{image_name}"


def get_last_commit_sha(api_url, token, branch):

    response = requests.get(f"{api_url}/git/ref/heads/{branch}", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        print("Error fetching the branch reference:", response.text)
        exit()

    last_commit_sha = response.json()["object"]["sha"]

    return last_commit_sha


def get_last_commit_tree(api_url, token, last_commit_sha):

    response = requests.get(f"{api_url}/git/commits/{last_commit_sha}", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        print("Error fetching the latest commit:", response.text)
        exit()

    base_tree_sha = response.json()["tree"]["sha"]

    return base_tree_sha


def create_imgs_blobs(api_url, token, file_names, base64_imgs):

    blobs = []

    for base64_img, file_name in zip(base64_imgs, file_names):

        blob_response = requests.post(
            f"{api_url}/git/blobs",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"content": base64_img, "encoding": "base64"},
        )

        if blob_response.status_code != 201:
            print(f"Error creating blob for {file_name}:", blob_response.text)
            exit()

        blob_sha = blob_response.json()["sha"]
        blobs.append({"path": file_name, "mode": "100644", "type": "blob", "sha": blob_sha})

    return blobs


def create_new_tree(api_url, token, base_tree_sha, blobs):

    tree_response = requests.post(
        f"{api_url}/git/trees",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"base_tree": base_tree_sha, "tree": blobs},
    )
    if tree_response.status_code != 201:
        print("Error creating the tree:", tree_response.text)
        exit()

    new_tree_sha = tree_response.json()["sha"]

    return new_tree_sha


def create_new_commit(api_url, token, last_commit_sha, new_tree_sha):

    commit_message = "Uploading multiple images in a single commit"
    commit_response = requests.post(
        f"{api_url}/git/commits",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"message": commit_message, "tree": new_tree_sha, "parents": [last_commit_sha]},
    )
    if commit_response.status_code != 201:
        print("Error creating the commit:", commit_response.text)
        exit()

    new_commit_sha = commit_response.json()["sha"]

    return new_commit_sha


def update_branch(api_url, token, branch, new_commit_sha):

    update_ref_response = requests.patch(
        f"{api_url}/git/refs/heads/{branch}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"sha": new_commit_sha},
    )
    if update_ref_response.status_code != 200:
        print("Error updating the branch reference:", update_ref_response.text)
        exit()
    else:
        print("Images Uploaded")


def upload_multiple_files_to_github(file_names, base64_imgs):

    owner, token, repo, branch = get_repo_config()

    # Base URL for the GitHub API
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    last_commit_sha = get_last_commit_sha(api_url, token, branch)
    base_tree_sha = get_last_commit_tree(api_url, token, last_commit_sha)
    blobs = create_imgs_blobs(api_url, token, file_names, base64_imgs)
    new_tree_sha = create_new_tree(api_url, token, base_tree_sha, blobs)
    new_commit_sha = create_new_commit(api_url, token, last_commit_sha, new_tree_sha)
    update_branch(api_url, token, branch, new_commit_sha)


def upload_file_to_github(file, file_name):

    # Config
    owner, token, repo, branch = get_repo_config()
    remote_file_name = f"{FOLDER}/{file_name}"

    # Read file content and encode it in Base64
    file_content = base64.b64encode(file.read().encode()).decode()

    # GitHub API URL for the file
    file_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{remote_file_name}"

    # Prepare the payload
    payload = {"message": f"Upload {file_name}", "content": file_content, "branch": branch}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Send the request to GitHub
    response = requests.put(file_url, json=payload, headers=headers)

    # Handle response
    if response.status_code == 201:
        print("File successfully uploaded")
        print("Public URL:", response.json()["content"]["download_url"])
        return response.json()["content"]["download_url"]
    else:
        print("Error uploading file:", response.status_code)
        print("Details:", response.json())
        return None


def format_sample(image, image_name: str, gt, owner: str, repo: str, branch: str, lang: str):

    base64_image = encode_image(image)
    url = compose_url(owner, repo, branch, image_name)
    
    if lang == "english":
        years = "'year_9', 'year_10', 'year_11', or 'year_12'"
        levels = "(9, 10, 11, or 12)"
        example_year = "year_9"
        rest_of_the_years = "year_10, year_11, or year_12"
    
    elif lang == "spanish":
        years = "'3_de_la_eso', '4_de_la_eso', '1_de_bachillerato', or '2_de_bachillerato'"
        levels = "(3º, 4º, 1º, or 2º)"
        example_year = "3_de_la_eso"
        rest_of_the_years = "4_de_la_eso, 1_de_bachillerato, or 2_de_bachillerato"
    
    else:
        years = [""]
        levels = ""
        example_year = ""
        rest_of_the_years = ""

    sample = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that extracts grades from students' transcripts of records.",
            },
            {
                "role": "user",
                "content": (
                    "Look at the image and extract:\n"
                    "- The subjects and their grades.\n"
                    f"- The level {levels} they correspond to.\n\n"
                    "You must return a SINGLE JSON object in the exact following format:\n\n"
                    "{\n"
                    f'  {example_year}: [  # Or {rest_of_the_years} as appropriate\n'
                    '    {"subject": "...", "grade": "..."},\n'
                    '    {"subject": "...", "grade": "..."}\n'
                    "  ]\n"
                    "}\n\n"
                    "DO NOT include any additional text, explanations, or comments. "
                    f"Use the key {years} based on what can be inferred from the image."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{url}"},
                    }
                ],
            },
            {"role": "assistant", "content": gt},
        ]
    }

    return sample, base64_image


def get_dataset_jsonl(decoded_ds_iterator, non_decoded_ds_iterator, lang: str):

    dataset_jsonl = []
    dataset_imgs = []
    dataset_imgs_names = []

    owner, _, repo, branch = get_repo_config()

    for i, (decoded_sample, non_decoded_sample) in enumerate(zip(decoded_ds_iterator, non_decoded_ds_iterator)):
        image, d_gt = get_sample_data(decoded_sample)
        image_name = get_sample_img_name(non_decoded_sample)
        sample, sample_img = format_sample(image, image_name, d_gt, owner, repo, branch, lang)
        dataset_jsonl.append(sample)
        dataset_imgs.append(sample_img)
        image_path = f"{FOLDER}/{SUBFOLDER}/{image_name}"
        dataset_imgs_names.append(image_path)

        if i + 1 >= TRAINING_SAMPLES:
            break

    return dataset_jsonl, dataset_imgs, dataset_imgs_names


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()

    subset_name = args.subset
    language = args.language
    
    set_folder(subset_name)

    decoded_ds_iterator = get_dataset_iterator(subset_name)
    non_decoded_ds_iterator = get_dataset_iterator(subset_name, True)

    dataset_jsonl, base64_imgs, base64_imgs_names = get_dataset_jsonl(decoded_ds_iterator, non_decoded_ds_iterator, language)

    ds_jsonl_file = save_dataset_jsonl("openai_finetuning_dataset.jsonl", dataset_jsonl)
    upload_file_to_github(ds_jsonl_file, "openai_finetuning_dataset.jsonl")
    upload_multiple_files_to_github(base64_imgs_names, base64_imgs)


if __name__ == "__main__":

    main()
