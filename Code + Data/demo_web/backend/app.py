from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from tokenizers import Tokenizer
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Translation API", version="1.0.0")

# CORS - Cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CẤU HÌNH MODEL ==========
MODEL_DIR = "./models/medical_translation"
MODEL_CHECKPOINT = f"{MODEL_DIR}/finetune_best.pt"
TOKENIZER_SRC = f"{MODEL_DIR}/tokenizer_en.json"
TOKENIZER_TGT = f"{MODEL_DIR}/tokenizer_vi.json"

model = None
tokenizer_src = None
tokenizer_tgt = None
device = None

# ========== IMPORT MODEL ARCHITECTURE ==========
# Thêm đường dẫn model directory vào sys.path để import
sys.path.insert(0, MODEL_DIR)

try:
    from model import build_transformer
    from config import get_config
    logger.info("✅ Imported build_transformer and get_config from model.py")
except ImportError as e:
    logger.error(f"❌ Could not import: {e}")
    build_transformer = None
    get_config = None

# ========== BEAM SEARCH FUNCTION ==========
def beam_search_translate(
    model,
    sentence,
    tokenizer_src,
    tokenizer_tgt,
    beam_size=4,
    max_len=100
):
    """
    Beam search translation - từ Kaggle notebook của bạn
    """
    model.eval()
    device = next(model.parameters()).device

    sos = tokenizer_tgt.token_to_id("[SOS]")
    eos = tokenizer_tgt.token_to_id("[EOS]")

    # -------- Encode source --------
    src_ids = tokenizer_src.encode(sentence).ids
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool).to(device)

    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

    # beam = (tokens, log_prob)
    beams = [([sos], 0.0)]

    for _ in range(max_len):
        new_beams = []

        for tokens, score in beams:
            # nếu đã kết thúc thì giữ nguyên
            if tokens[-1] == eos:
                new_beams.append((tokens, score))
                continue

            tgt = torch.tensor(tokens).unsqueeze(0).to(device)
            tgt_mask = torch.tril(
                torch.ones(1, tgt.size(1), tgt.size(1), device=device)
            ).bool()

            with torch.no_grad():
                out = model.decode(encoder_output, src_mask, tgt, tgt_mask)
                logits = model.project(out[:, -1]).squeeze(0)
                log_probs = torch.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = topk.indices[i].item()
                next_score = score + topk.values[i].item()
                new_beams.append((tokens + [next_token], next_score))

        # nếu beam chết thì dừng
        if len(new_beams) == 0:
            break

        # giữ top beam
        beams = sorted(new_beams, key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = beams[:beam_size]

        # stop sớm nếu tất cả đều EOS
        if all(b[0][-1] == eos for b in beams):
            break

    if len(beams) == 0:
        return ""

    best_tokens = beams[0][0]

    # bỏ SOS và EOS
    return tokenizer_tgt.decode(best_tokens[1:-1])

def load_model():
    """Load model và tokenizers khi khởi động server"""
    global model, tokenizer_src, tokenizer_tgt, device
    
    try:
        logger.info("=" * 60)
        logger.info("🔄 LOADING MODEL AND TOKENIZERS")
        logger.info("=" * 60)
        
        # Kiểm tra files tồn tại
        if not os.path.exists(MODEL_DIR):
            logger.error(f"❌ Model directory not found: {MODEL_DIR}")
            return False
        
        logger.info(f"📁 Model directory: {MODEL_DIR}")
        logger.info("📂 Files in directory:")
        for f in os.listdir(MODEL_DIR):
            logger.info(f"   - {f}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"📱 Using device: {device}")
        
        # Load tokenizers
        logger.info("🔄 Loading tokenizers...")
        if not os.path.exists(TOKENIZER_SRC):
            logger.error(f"❌ Source tokenizer not found: {TOKENIZER_SRC}")
            return False
        if not os.path.exists(TOKENIZER_TGT):
            logger.error(f"❌ Target tokenizer not found: {TOKENIZER_TGT}")
            return False
        
        tokenizer_src = Tokenizer.from_file(TOKENIZER_SRC)
        tokenizer_tgt = Tokenizer.from_file(TOKENIZER_TGT)
        logger.info(f"✅ Tokenizers loaded!")
        logger.info(f"   Source vocab size: {tokenizer_src.get_vocab_size()}")
        logger.info(f"   Target vocab size: {tokenizer_tgt.get_vocab_size()}")
        
        # Kiểm tra model checkpoint
        if not os.path.exists(MODEL_CHECKPOINT):
            logger.error(f"❌ Model checkpoint not found: {MODEL_CHECKPOINT}")
            return False
        
        # Kiểm tra build_transformer function
        if build_transformer is None:
            logger.error("❌ build_transformer function not found!")
            logger.error("⚠️  Check models/medical_translation/model.py")
            return False
        
        # Load checkpoint
        logger.info(f"🔄 Loading checkpoint from {MODEL_CHECKPOINT}...")
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        
        # Get config from config.py
        config = get_config()
        logger.info(f"📋 Config loaded: seq_len={config['seq_len']}, d_model={config['d_model']}")
        
        # Khởi tạo model với build_transformer (giống Kaggle notebook)
        logger.info("🔄 Initializing model with build_transformer...")
        model = build_transformer(
            src_vocab_size=tokenizer_src.get_vocab_size(),
            tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
            src_seq_len=config['seq_len'],
            d_model=config['d_model']
        ).to(device)
        
        # Load state dict
        logger.info("🔄 Loading model weights...")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Xử lý prefix _orig_mod. (từ torch.compile)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("✅ Model loaded successfully!")
        
        # Test model
        logger.info("🧪 Testing model with sample translation...")
        try:
            test_result = beam_search_translate(
                model,
                "test",
                tokenizer_src,
                tokenizer_tgt,
                beam_size=2,
                max_len=20
            )
            logger.info(f"✅ Test translation: 'test' → '{test_result}'")
        except Exception as e:
            logger.warning(f"⚠️  Test translation failed: {e}")
        
        logger.info("=" * 60)
        logger.info("✅ MODEL LOADED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model khi khởi động
@app.on_event("startup")
async def startup_event():
    success = load_model()
    if success:
        logger.info("🚀 Server started successfully with model loaded!")
    else:
        logger.warning("⚠️ Server started but model failed to load. Check model path!")

# ========== API SCHEMA ==========
class TranslationRequest(BaseModel):
    text: str
    source: str = "en"
    target: str = "vi"
    beam_size: int = 4  # Cho phép tùy chỉnh beam size

class TranslationResponse(BaseModel):
    translatedText: str
    source: str
    target: str
    modelLoaded: bool

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Translation API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/api/health")
async def health_check():
    """Check if API and model are ready"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate medical text from English to Vietnamese using beam search
    
    Example:
    ```json
    {
        "text": "The patient shows signs of acute respiratory distress",
        "source": "en",
        "target": "vi",
        "beam_size": 4
    }
    ```
    """
    # Kiểm tra model đã load chưa
    if model is None or tokenizer_src is None or tokenizer_tgt is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and ensure model files are in place."
        )
    
    # Validate input
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Validate beam_size
    if request.beam_size < 1 or request.beam_size > 10:
        raise HTTPException(status_code=400, detail="beam_size must be between 1 and 10")
    
    try:
        logger.info(f"📝 Translating: '{request.text[:50]}...' (beam_size={request.beam_size})")
        
        # Dịch bằng beam search
        translated_text = beam_search_translate(
            model,
            request.text,
            tokenizer_src,
            tokenizer_tgt,
            beam_size=request.beam_size,
            max_len=100
        )
        
        logger.info(f"✅ Translation complete: '{translated_text[:50]}...'")
        
        return TranslationResponse(
            translatedText=translated_text,
            source=request.source,
            target=request.target,
            modelLoaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Translation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Translation failed: {str(e)}"
        )

# ========== RUN SERVER ==========
if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("🚀 STARTING MEDICAL TRANSLATION API")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
