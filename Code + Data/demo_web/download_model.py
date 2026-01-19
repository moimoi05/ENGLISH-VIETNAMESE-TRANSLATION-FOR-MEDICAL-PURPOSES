# Script tự động download model từ Kaggle
import os
import subprocess
import sys

def check_kaggle_setup():
    """Kiểm tra Kaggle API đã setup chưa"""
    print("🔍 Checking Kaggle API setup...")
    
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle API token not found!")
        print("\n📝 To setup:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New Token'")
        print("3. Download kaggle.json")
        print(f"4. Copy to: {kaggle_dir}")
        return False
    
    print("✅ Kaggle API token found!")
    return True

def download_model():
    """Download model từ Kaggle notebook"""
    print("\n📥 Downloading model from Kaggle...")
    print("Notebook: https://www.kaggle.com/code/moimoi05/inference-dichmay")
    
    # Tạo folder nếu chưa có
    model_dir = "./models/medical_translation"
    os.makedirs(model_dir, exist_ok=True)
    
    # Command download
    cmd = f"kaggle kernels output moimoi05/inference-dichmay -p {model_dir}"
    
    try:
        print(f"\n🔄 Running: {cmd}")
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        print("✅ Download successful!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Download failed!")
        print(f"Error: {e.stderr}")
        print("\n💡 Alternative solutions:")
        print("1. Check Kaggle API token is valid")
        print("2. Or download manually from: https://www.kaggle.com/code/moimoi05/inference-dichmay")
        print("   - Click 'Output' tab")
        print("   - Download all files")
        print(f"   - Extract to: {os.path.abspath(model_dir)}")
        return False

def list_downloaded_files():
    """List các files đã download"""
    model_dir = "./models/medical_translation"
    
    print(f"\n📁 Files in {model_dir}:")
    
    if not os.path.exists(model_dir):
        print("  (Folder not found)")
        return False
    
    files = os.listdir(model_dir)
    
    if not files:
        print("  (Empty folder)")
        return False
    
    for file in files:
        file_path = os.path.join(model_dir, file)
        size = os.path.getsize(file_path)
        size_mb = size / (1024 * 1024)
        print(f"  ✓ {file} ({size_mb:.2f} MB)")
    
    # Kiểm tra các files cần thiết
    print("\n✅ Checking required files:")
    required_files = ["config.json", "tokenizer_config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors", "model.pth"]
    
    has_config = any(f in files for f in required_files)
    has_model = any(f in files for f in model_files)
    
    if has_config:
        print("  ✓ Config files found")
    else:
        print("  ⚠ Config files missing")
    
    if has_model:
        print("  ✓ Model weights found")
    else:
        print("  ⚠ Model weights missing")
    
    return has_config and has_model

def main():
    print("=" * 60)
    print("🚀 DOWNLOAD MODEL FROM KAGGLE")
    print("=" * 60)
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n⛔ Please setup Kaggle API first!")
        sys.exit(1)
    
    # Download model
    success = download_model()
    
    if not success:
        print("\n⛔ Download failed! Please download manually.")
        sys.exit(1)
    
    # List files
    has_required = list_downloaded_files()
    
    if has_required:
        print("\n" + "=" * 60)
        print("✅ MODEL DOWNLOADED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📝 Next steps:")
        print("1. Make sure you have requirements.txt installed:")
        print("   pip install -r requirements.txt")
        print("\n2. Run the backend server:")
        print("   python app.py")
        print("\n3. Test the API:")
        print("   http://localhost:8000/api/health")
    else:
        print("\n⚠️ WARNING: Some required files may be missing!")
        print("Please check the model folder manually.")

if __name__ == "__main__":
    main()
