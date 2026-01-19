from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    # Map lang code to dataset column name
    col_map = {'en': 'EnglishSentences', 'vi': 'VietnameseSentences'}
    col_name = col_map.get(lang, lang)
    
    for item in ds:
        if 'translation' in item:
            yield item['translation'][lang]
        else:
            yield item[col_name]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        # Sử dụng BPE thay vì WordLevel
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # BpeTrainer cần chỉ định vocab_size (VD: 30000 là chuẩn phổ biến)
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=30000, min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
         
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
    
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:

        src_target_pair = self.ds[index]
        
        # --- OPTIMIZATION: Use pre-tokenized IDs if available ---
        if 'src_ids' in src_target_pair and 'tgt_ids' in src_target_pair:
            enc_input_tokens = src_target_pair['src_ids']
            dec_input_tokens = src_target_pair['tgt_ids']
            src_text = src_target_pair.get('src_text', "")  # Optional
            tgt_text = src_target_pair.get('tgt_text', "")  # Optional
        else:
            # Fallback to slow on-the-fly tokenization
            if 'translation' in src_target_pair:
                src_text = src_target_pair['translation'][self.src_lang]
                tgt_text = src_target_pair['translation'][self.tgt_lang]
            else:
                src_map = {'en': 'EnglishSentences', 'vi': 'VietnameseSentences'}
                tgt_map = {'en': 'EnglishSentences', 'vi': 'VietnameseSentences'}
                src_col = src_map.get(self.src_lang, self.src_lang)
                tgt_col = tgt_map.get(self.tgt_lang, self.tgt_lang)
                src_text = src_target_pair[src_col]
                tgt_text = src_target_pair[tgt_col]

            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # --------------------------------------------------------

        # Dynamic Padding: Không padding tại đây, chỉ thêm SOS/EOS
        
        # Add SOS and EOS tokens to encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],            
            dim=0,
        )

        # Add SOS token to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        # Add EOS token to decoder label
        lable = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],
            dim=0,
        )

        # Không tạo mask ở đây nữa, sẽ tạo trong collate_fn hoặc model
        # Trả về tensors có độ dài khác nhau
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": lable,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causual_mask(size):
    mask = torch.triu(torch.ones(1,size, size),diagonal=1).type(torch.int)
    return mask == 0

