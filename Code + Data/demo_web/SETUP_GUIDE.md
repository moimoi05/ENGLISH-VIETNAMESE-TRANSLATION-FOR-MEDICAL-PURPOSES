# 🚀 Hướng Dẫn Chạy Medical Translation AI trên Localhost

Hướng dẫn chi tiết để triển khai và chạy ứng dụng dịch y tế AI trên máy mới.

---

## 📋 Yêu Cầu Hệ Thống

### Phần mềm cần cài đặt:

1. **Python 3.8+** (khuyến nghị Python 3.10 trở lên)
   - Download: https://www.python.org/downloads/
   - Kiểm tra: `python --version`

2. **Node.js 16+** (khuyến nghị Node.js 18 LTS)
   - Download: https://nodejs.org/
   - Kiểm tra: `node --version` và `npm --version`

3. **Git** (để clone project)
   - Download: https://git-scm.com/
   - Kiểm tra: `git --version`

4. **CUDA Toolkit** (nếu có GPU NVIDIA - không bắt buộc)
   - Download: https://developer.nvidia.com/cuda-downloads
   - Nếu không có GPU, PyTorch sẽ tự động dùng CPU

### Yêu cầu phần cứng:
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: NVIDIA GPU với CUDA support (không bắt buộc, có thể chạy trên CPU)
- **Ổ cứng**: ~5GB dung lượng trống

---

## 📁 Cấu Trúc Project

```
demo_web/
├── backend/                          # Backend FastAPI server
│   ├── models/
│   │   └── medical_translation/     # Model files
│   │       ├── config.py            # Config file từ Kaggle
│   │       ├── model.py             # Model architecture từ Kaggle
│   │       ├── finetune_best.pt     # Checkpoint đã train
│   │       ├── tokenizer_en.json    # Tokenizer tiếng Anh
│   │       └── tokenizer_vi.json    # Tokenizer tiếng Việt
│   ├── app.py                       # FastAPI application
│   └── requirements.txt             # Python dependencies
│
├── src/                             # Frontend React source
│   ├── components/                  # React components
│   │   ├── Translator.jsx          # Translation interface
│   │   └── History.jsx             # Translation history
│   ├── services/
│   │   └── translationApi.js       # API service
│   ├── App.jsx                     # Main app component
│   └── main.jsx                    # Entry point
│
├── package.json                     # Node.js dependencies
└── vite.config.js                  # Vite configuration
```

---

## 🔧 Bước 1: Clone/Copy Project

### Nếu có Git repository:
```bash
git clone <repository-url>
cd demo_web
```

### Nếu copy từ máy khác:
- Copy toàn bộ folder `demo_web` vào máy mới
- Đảm bảo có đầy đủ 5 file trong `backend/models/medical_translation/`:
  - `config.py`
  - `model.py`
  - `finetune_best.pt`
  - `tokenizer_en.json`
  - `tokenizer_vi.json`

---

## 🐍 Bước 2: Setup Backend (Python)

### 2.1. Tạo Python Virtual Environment

```bash
# Windows
cd backend
python -m venv venv
venv\Scripts\activate

# Linux/Mac
cd backend
python3 -m venv venv
source venv/bin/activate
```

**Lưu ý**: Sau khi activate, bạn sẽ thấy `(venv)` ở đầu dòng lệnh.

### 2.2. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

**Thời gian cài đặt**: ~5-10 phút (tùy tốc độ mạng và có GPU hay không)

**Các package chính**:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch (deep learning)
- `tokenizers` - Tokenization library
- `python-multipart` - Form data handling

### 2.3. Kiểm tra cài đặt

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Kết quả mong đợi**:
```
PyTorch version: 2.x.x
CUDA available: True  # hoặc False nếu không có GPU
```

### 2.4. Chạy Backend Server

```bash
python app.py
```

**Kết quả thành công**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:__main__:✅ MODEL LOADED SUCCESSFULLY!
INFO:__main__:🚀 Server started successfully with model loaded!
```

**Backend API endpoints**:
- `http://localhost:8000` - API root
- `http://localhost:8000/api/health` - Health check
- `http://localhost:8000/api/translate` - Translation endpoint
- `http://localhost:8000/docs` - API documentation (Swagger UI)

**Lưu ý**: 
- Backend sẽ chạy trên port **8000**
- Để dừng server: `Ctrl+C`
- Nếu port 8000 bị chiếm: Tìm process đang dùng `netstat -ano | findstr :8000` và kill nó

---

## ⚛️ Bước 3: Setup Frontend (React)

### 3.1. Cài đặt Node.js Dependencies

Mở terminal/cmd **mới** (giữ backend chạy):

```bash
# Từ thư mục gốc demo_web
npm install
```

**Thời gian cài đặt**: ~2-5 phút

**Các package chính**:
- `react` & `react-dom` - React framework
- `vite` - Build tool
- `tailwindcss` - CSS framework

### 3.2. Chạy Frontend Development Server

```bash
npm run dev
```

**Kết quả thành công**:
```
VITE v5.x.x  ready in 437 ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

**Frontend sẽ chạy trên**:
- Port **3000** (hoặc 3001 nếu 3000 bị chiếm)
- Auto-reload khi có thay đổi code

**Lưu ý**:
- Để dừng: `Ctrl+C`
- CORS đã được cấu hình cho port 3000 và 3001

---

## ✅ Bước 4: Test Ứng Dụng

### 4.1. Mở trình duyệt

Truy cập: **http://localhost:3000** (hoặc port mà Vite hiển thị)

### 4.2. Test chức năng dịch

1. **Nhập văn bản tiếng Anh** vào ô bên trái
   - Ví dụ: `The patient has diabetes mellitus`
   - Chữ cái đầu tiên sẽ tự động viết hoa

2. **Dịch văn bản**:
   - Cách 1: Bấm nút **"Dịch văn bản"**
   - Cách 2: Bấm **Enter** (Shift+Enter để xuống dòng)

3. **Kiểm tra kết quả** ở ô bên phải

4. **Xem lịch sử dịch** ở phần dưới
   - Click vào mục lịch sử để load lại
   - Bấm **"Xóa lịch sử"** để xóa tất cả

### 4.3. Test API trực tiếp

```bash
# Test health check
curl http://localhost:8000/api/health

# Test translation (PowerShell)
Invoke-RestMethod -Uri "http://localhost:8000/api/translate" -Method Post -ContentType "application/json" -Body '{"text":"The patient has fever","source":"en","target":"vi"}'
```

---

## 🐛 Xử Lý Lỗi Thường Gặp

### Backend Issues

#### 1. **Port 8000 đã được sử dụng**

**Lỗi**: `error while attempting to bind on address ('0.0.0.0', 8000)`

**Giải quyết**:
```bash
# Tìm process đang dùng port 8000
netstat -ano | findstr :8000

# Kill process (thay <PID> bằng số process)
taskkill /PID <PID> /F
```

#### 2. **ModuleNotFoundError**

**Lỗi**: `ModuleNotFoundError: No module named 'fastapi'`

**Giải quyết**:
```bash
# Đảm bảo đã activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Cài lại dependencies
pip install -r requirements.txt
```

#### 3. **Model không load được**

**Lỗi**: `❌ Error loading model`

**Kiểm tra**:
```bash
# Verify file tồn tại
dir backend\models\medical_translation\  # Windows
ls backend/models/medical_translation/   # Linux/Mac

# Phải có 5 files:
# - config.py
# - model.py
# - finetune_best.pt (checkpoint)
# - tokenizer_en.json
# - tokenizer_vi.json
```

#### 4. **CUDA/GPU errors**

**Lỗi**: CUDA-related errors

**Giải quyết**:
- Model sẽ tự động dùng CPU nếu không có GPU
- Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`
- Nếu muốn force CPU: Sửa `app.py` dòng `device = torch.device("cpu")`

### Frontend Issues

#### 1. **Port 3000 đã được sử dụng**

**Vite sẽ tự động dùng port khác** (3001, 3002...)

Nếu muốn force port:
```bash
npm run dev -- --port 3005
```

#### 2. **CORS errors**

**Lỗi**: `Failed to fetch` hoặc CORS policy error

**Giải quyết**:
- Kiểm tra backend đang chạy: http://localhost:8000/api/health
- Đảm bảo frontend chạy đúng port (3000 hoặc 3001)
- Port được config trong `backend/app.py`:
  ```python
  allow_origins=["http://localhost:3000", "http://localhost:3001"]
  ```

#### 3. **npm install fails**

**Giải quyết**:
```bash
# Xóa cache và cài lại
rm -rf node_modules package-lock.json  # Linux/Mac
rmdir /s node_modules & del package-lock.json  # Windows

npm install
```

---

## 🎯 Tips & Best Practices

### Performance

1. **GPU vs CPU**:
   - GPU (CUDA): ~2-5 giây/câu
   - CPU: ~10-30 giây/câu
   - Model tự động detect và dùng GPU nếu có

2. **Beam Search**:
   - `beam_size=4` (mặc định) - cân bằng tốc độ/chất lượng
   - Tăng beam_size → chất lượng tốt hơn, chậm hơn
   - Giảm beam_size → nhanh hơn, chất lượng kém hơn

### Development

1. **Hot reload**:
   - Frontend: Vite tự động reload khi sửa code
   - Backend: Cần restart server sau khi sửa code

2. **Debugging**:
   - Backend logs: Xem terminal chạy `python app.py`
   - Frontend logs: F12 → Console trong browser
   - API test: http://localhost:8000/docs

3. **Model files**:
   - Không commit `finetune_best.pt` vào Git (file quá lớn ~300MB)
   - Upload lên Google Drive/Dropbox để chia sẻ
   - Sử dụng Git LFS nếu cần version control

---

## 📦 Production Deployment

### Build Frontend

```bash
npm run build
```

Tạo folder `dist/` với static files tối ưu.

### Serve Production

```bash
npm run preview
```

Hoặc dùng backend để serve frontend:
```bash
# Thêm vào backend/app.py
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="../dist", html=True), name="frontend")
```

---

## 📞 Hỗ Trợ

### Logs quan trọng

**Backend startup**:
```
✅ Imported build_transformer and get_config from model.py
✅ Tokenizers loaded!
✅ Model loaded successfully!
✅ MODEL LOADED SUCCESSFULLY!
```

**Frontend startup**:
```
VITE v5.x.x ready in 437 ms
➜  Local: http://localhost:3000/
```

### Kiểm tra status

```bash
# Backend health
curl http://localhost:8000/api/health

# Hoặc mở browser: http://localhost:8000/docs
```

---

## 🎓 Tóm Tắt Commands

```bash
# ==== BACKEND ====
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py

# ==== FRONTEND (terminal mới) ====
npm install
npm run dev

# ==== TEST ====
# Mở browser: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

---

## ✨ Chúc Bạn Thành Công!

Nếu gặp vấn đề không có trong hướng dẫn này, kiểm tra:
1. Logs trong terminal backend
2. Console (F12) trong browser
3. Network tab để xem API requests

**Good luck!** 🚀
