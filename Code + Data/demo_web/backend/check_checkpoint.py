# Script kiểm tra checkpoint
import torch

checkpoint_path = "./models/medical_translation/finetune_best.pt"

print("🔍 Đang kiểm tra checkpoint...")
print(f"📁 File: {checkpoint_path}\n")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"✅ Load thành công!")
    print(f"📦 Type: {type(checkpoint)}\n")
    
    if isinstance(checkpoint, dict):
        print("📋 Keys trong checkpoint:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  - {key}: dict với {len(checkpoint[key])} items")
            else:
                print(f"  - {key}: {type(checkpoint[key])}")
        
        # Kiểm tra có model hoàn chỉnh không
        if 'model' in checkpoint:
            print(f"\n✅ Có key 'model': {type(checkpoint['model'])}")
        
        if 'model_state_dict' in checkpoint:
            print(f"✅ Có key 'model_state_dict' với {len(checkpoint['model_state_dict'])} layers")
            print("\n📝 Một số layers đầu tiên:")
            for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:5]):
                print(f"  - {key}")
    
    elif hasattr(checkpoint, 'state_dict'):
        print("✅ Checkpoint là model hoàn chỉnh (có method state_dict)")
        print(f"📝 Type: {type(checkpoint)}")
    
    else:
        print("⚠️ Checkpoint có format đặc biệt")
        
except Exception as e:
    print(f"❌ Lỗi: {e}")
    import traceback
    traceback.print_exc()
