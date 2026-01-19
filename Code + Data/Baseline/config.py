from pathlib import Path


def get_config():
    return {
        "batch_size": 128,  
        "accumulation_steps": 4,
        "num_epochs": 20,
        "lr":10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "vi",
        "model_folder": "/content/drive/MyDrive/DL/Transformer from Scratch/weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    model_folder_path = Path(model_folder)
    if not model_folder_path.is_absolute():
        model_folder_path = Path('.') / model_folder_path
    return str(model_folder_path / model_filename)

def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_folder_path = Path(model_folder)
    if not model_folder_path.is_absolute():
        model_folder_path = Path('.') / model_folder_path
    files = list(model_folder_path.glob(f"{model_basename}*.pt"))
    if len(files) == 0:
        return None
    files.sort()
    return str(files[-1])
