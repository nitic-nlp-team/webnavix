from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="McGill-NLP/WebLINX-full",
    repo_type="dataset",
    local_dir="./wl_data",
    ignore_patterns=["**/video.mp4"],
    resume_download=True,
)
