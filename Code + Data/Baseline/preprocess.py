from datasets import load_dataset, DatasetDict
from config import get_config
from dataset import get_or_build_tokenizer
from pathlib import Path

def process_pipeline():
    config = get_config()
    dataset_cache_path = Path("/content/drive/MyDrive/DL/Transformer from Scratch/dataset_cache_vi")
    
    if dataset_cache_path.exists():
        print(f"Cache already exists at {dataset_cache_path}. Delete it if you want to re-process.")
        return
    
    print("Loading raw dataset...")
    # Load dataset
    ds_raw = load_dataset('HelloWorld2307/eng_viet_translation', split='train')
    
    # Select columns
    cols_to_keep = ['EnglishSentences', 'VietnameseSentences']
    ds_raw = ds_raw.select_columns(cols_to_keep)
    
    # Build tokenizers
    print("Building tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Pre-tokenization
    print("Pre-tokenizing dataset...")
    src_col_map = {'en': 'EnglishSentences', 'vi': 'VietnameseSentences'}
    tgt_col_map = {'en': 'EnglishSentences', 'vi': 'VietnameseSentences'}
    src_col = src_col_map.get(config['lang_src'], config['lang_src'])
    tgt_col = tgt_col_map.get(config['lang_tgt'], config['lang_tgt'])
    
    def process_data(examples):
        if 'translation' in examples:
             src_text = [x[config['lang_src']] for x in examples['translation']]
             tgt_text = [x[config['lang_tgt']] for x in examples['translation']]
        else:
             src_text = examples[src_col]
             tgt_text = examples[tgt_col]
        
        src_encodings = tokenizer_src.encode_batch(src_text)
        tgt_encodings = tokenizer_tgt.encode_batch(tgt_text)
        
        return {
            "src_ids": [x.ids for x in src_encodings],
            "tgt_ids": [x.ids for x in tgt_encodings],
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
    ds_raw = ds_raw.map(process_data, batched=True, remove_columns=ds_raw.column_names)
    
    # Filter
    print("Filtering long sentences...")
    def filter_length(example):
        return len(example['src_ids']) <= config['seq_len'] - 2 and \
               len(example['tgt_ids']) <= config['seq_len'] - 2
    
    ds_raw = ds_raw.filter(filter_length)
    print(f"Dataset size after filtering: {len(ds_raw)}")
    
    # --- SPLIT TRAIN/VALIDATION/TEST (98/1/1) ---
    print("Splitting dataset into Train (98%), Validation (1%), Test (1%)...")
    
    # Bước 1: Tách 2% ra làm tập tạm (Train=98%, Temp=2%)
    ds_train_temp = ds_raw.train_test_split(test_size=0.02, seed=42)
    
    # Bước 2: Chia đôi tập Temp (mỗi bên 1% tổng gốc) thành Validation và Test
    ds_val_test = ds_train_temp['test'].train_test_split(test_size=0.5, seed=42)
    
    # Gom lại thành DatasetDict
    ds_final = DatasetDict({
        'train': ds_train_temp['train'],
        'validation': ds_val_test['train'],
        'test': ds_val_test['test']
    })
    
    # Save
    print(f"Saving processed dataset to {dataset_cache_path}...")
    ds_final.save_to_disk(str(dataset_cache_path))
    print("Done! Structure: ", ds_final)

if __name__ == "__main__":
    process_pipeline()
