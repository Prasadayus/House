from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
  path_or_fileobj="model/gbr_house_model.pkl",
  path_in_repo="gbr_house_model.pkl",
  repo_id="PAyus77/house-price-model"
)
