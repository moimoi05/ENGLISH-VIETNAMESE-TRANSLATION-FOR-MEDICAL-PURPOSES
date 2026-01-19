import torch
from model import build_transformer
from .dataset import get_dataloaders
from .trainer import train_model
from .config import BASE_MODEL_PATH

def train_medical_experiment(
    N,
    d_model,
    d_ff,
    h,
    dropout,
    epsilon_ls,
    train_steps,
    batch_size=32,
    warmup_steps=4000,
    device=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Nhận thêm train_sampler
    train_dl, val_dl, test_dl, tok_src, tok_tgt, train_sampler = get_dataloaders(batch_size)

    print(f"🛠️ Building Transformer with N={N}, d_model={d_model}...")
    model = build_transformer(
        tok_src.get_vocab_size(),
        tok_tgt.get_vocab_size(),
        350,
        d_model,
        N,
        h,
        dropout,
        d_ff,
    ).to(device)

    # --- SMART CHECKPOINT LOADING ---
    print(f"📥 Loading weights from {BASE_MODEL_PATH}")
    try:
        checkpoint = torch.load(BASE_MODEL_PATH, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            
            # 1. FIX TYPO: liner -> linear
            if "liner" in new_k:
                new_k = new_k.replace("liner", "linear")
            
            # 2. REMOVE PREFIX: _orig_mod
            new_k = new_k.replace("_orig_mod.", "")

            # 3. FILTER LAYERS: Nếu checkpoint có N=6 mà ta chỉ init N=4
            # Key dạng: encoder.layers.5.xyz...
            parts = new_k.split('.')
            if "layers" in parts:
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    if layer_idx >= N:
                        continue  # Bỏ qua layer này vì model mới nhỏ hơn
                except ValueError:
                    pass

            # 4. FIX LAYERNORM SHAPE: [1] -> [512]
            # Checkpoint cũ LayerNorm bị sai shape [1], model mới cần [d_model]
            if "norm" in new_k and (v.shape == torch.Size([1])):
                # Duplicate giá trị duy nhất đó ra thành vector [d_model]
                # Điều này giữ nguyên hành vi cũ (tất cả feature scale như nhau) để model không bị sốc
                v = v.repeat(d_model)

            new_state_dict[new_k] = v

        # Load với strict=False để bỏ qua những key không khớp nhỏ nhặt
        # missing_keys: Các key model mới có mà checkpoint không có (sẽ random init)
        # unexpected_keys: Các key checkpoint có mà model mới không cần
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("✅ Load weights thành công (với điều chỉnh)!")
        print(f"   - Missing keys (sẽ init mới): {len(msg.missing_keys)}")
        print(f"   - Unexpected keys (đã bỏ): {len(msg.unexpected_keys)}")

    except FileNotFoundError:
        print("⚠️ Không tìm thấy file weights. Sẽ train từ đầu (Scratch).")
    except Exception as e:
        print(f"⚠️ Lỗi khi load weights: {e}")
        print("➡️ Tiếp tục train từ đầu (Scratch)...")

    # Optimization: Compile model (Tăng tốc đáng kể trên A100 với PyTorch 2.x)
    print("🚀 Compiling model for A100...")
    model = torch.compile(model)

    # Hàm train_model giờ trả về history dict
    history = train_model(
        model,
        train_dl,
        val_dl,
        test_dl,
        tok_tgt,
        train_steps,
        epsilon_ls,
        d_model,
        warmup_steps,
        device,
        train_sampler=train_sampler,
        accumulation_steps=4,
    )

    return history
