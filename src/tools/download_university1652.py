from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="layumi/university-1652",
    repo_type="dataset",
    local_dir="data/university-1652",
)
print("downloaded metadata to data/university-1652")
print("note: the HuggingFace mirror does NOT include image files.")
