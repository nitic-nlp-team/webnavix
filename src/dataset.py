import time

from huggingface_hub import snapshot_download

attempt = 1
while True:
    try:
        print(f"Attempt {attempt} for dataset 'McGill-NLP/WebLINX-full'.")  # noqa: T201

        snapshot_download(
            repo_id="McGill-NLP/WebLINX-full",
            repo_type="dataset",
            local_dir="./wl_data",
            ignore_patterns=["**/video.mp4"],
            resume_download=True,
            max_workers=16,
        )

        print("Download successful!")  # noqa: T201
        break

    except Exception as e:  # noqa: BLE001
        print(f"Error occurred: {e}")  # noqa: T201

        print("Retrying in 5 seconds...")  # noqa: T201
        time.sleep(5)
        attempt += 1
