import torch
from torch.utils.data import Dataset
from dataset import causual_mask  # Import hàm mask từ model.py

class MedTranslationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer_src, tokenizer_tgt, seq_len):
        self.ds = hf_dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len

        # Cache các token ID đặc biệt
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # 1. Lấy text
        src_text = self.ds[idx]["Eng"]
        tgt_text = self.ds[idx]["Vie"]

        # 2. Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 3. Tính số lượng PAD cần thêm để đạt độ dài cố định seq_len (350)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 cho SOS và EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 cho SOS

        # Nếu câu quá dài thì cắt bớt (Truncate)
        if enc_num_padding_tokens < 0:
            enc_input_tokens = enc_input_tokens[:enc_num_padding_tokens]
            enc_num_padding_tokens = 0
        
        if dec_num_padding_tokens < 0:
            dec_input_tokens = dec_input_tokens[:dec_num_padding_tokens]
            dec_num_padding_tokens = 0

        # 4. Tạo Tensor với Padding (Quan trọng nhất để fix lỗi stack)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        # Đảm bảo kích thước luôn cố định là seq_len (350)
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # 5. Tạo Mask
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causual_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }