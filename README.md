# Medical Translation AI - Dịch Máy Chuyên Ngành Y Tế Anh-Việt

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Hệ thống dịch máy Neural Machine Translation (NMT) chuyên ngành Y tế, xây dựng kiến trúc Transformer from scratch và Fine-tuning trên dữ liệu y khoa chuyên biệt.

---

## Thành Viên Nhóm 14

| Họ và Tên              |      Mã Sinh Viên     |
|------------------------|-----------------------|
| Nguyễn Phương Nam      | 23020406              |
| Nguyễn Trọng Hồng Phúc | 23020410              |
| Nguyễn Đình Quyền      | 23020422              |
| Trần Doãn Thắng        | 23020438              |

Vì dự án dùng Google Colab nên mọi code chi tiết đều ở trên Drive: [https://drive.google.com/drive/folders/1fMnFoO3o4NakaGuCvMUDYcD6Nnf7TR53?usp=sharing]
---
Nếu có thắc mắc, vui lòng liên hệ qua mail: [nnam.hp2005@gmail.com]
---

## Mục Lục

- [Giới Thiệu Dự Án](#-giới-thiệu-dự-án)
- [Đặt Vấn Đề Nghiên Cứu](#-đặt-vấn-đề-nghiên-cứu)
- [Mục Tiêu Nghiên Cứu](#-mục-tiêu-nghiên-cứu)
- [Kiến Trúc Hệ Thống](#️-kiến-trúc-hệ-thống)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Demo Web Application](#-demo-web-application)
- [Kết Quả Thử Nghiệm](#-kết-quả-thử-nghiệm)
- [Tài Liệu Tham Khảo](#-tài-liệu-tham-khảo)

---

## Giới Thiệu Dự Án

Dự án **Medical Translation AI** là một hệ thống dịch máy thần kinh (Neural Machine Translation) chuyên biệt cho lĩnh vực **Y tế**, thực hiện dịch từ **tiếng Anh sang tiếng Việt** với độ chính xác cao trên các thuật ngữ y khoa và ngữ cảnh lâm sàng phức tạp.

### Điểm Nổi Bật

- **Transformer from Scratch**: Xây dựng hoàn toàn kiến trúc Transformer từ đầu (không sử dụng thư viện có sẵn ở mức cao)
- **Two-Stage Training**: Huấn luyện Baseline trên dữ liệu phổ thông, sau đó Fine-tune trên dữ liệu y tế chuyên biệt
- **Domain-Specific**: Tối ưu hóa cho thuật ngữ y khoa, triệu chứng, chẩn đoán và điều trị
- **Web Application**: Giao diện web thân thiện với người dùng, hỗ trợ dịch real-time
- **Beam Search Decoding**: Sử dụng thuật toán Beam Search để tối ưu chất lượng dịch

---

##  Đặt Vấn Đề Nghiên Cứu

### Bối Cảnh

Trong kỷ nguyên toàn cầu hóa, **rào cản ngôn ngữ** là một trong những trở ngại lớn nhất trong việc tiếp cận tri thức, đặc biệt là trong lĩnh vực **Y học**. Các tài liệu y khoa cập nhật nhất, các nghiên cứu lâm sàng và hướng dẫn điều trị chuẩn mực thường được viết bằng tiếng Anh. 

Tại Việt Nam, nhu cầu dịch thuật tài liệu y tế là rất lớn để phục vụ cho:
-  Bác sĩ và nhân viên y tế
-  Sinh viên y khoa
-  Bệnh nhân và người nhà bệnh nhân

### Thách Thức

Các công cụ dịch phổ biến hiện nay (như Google Translate) thường hoạt động tốt trên ngôn ngữ đời sống nhưng gặp khó khăn với:
-  **Thuật ngữ chuyên ngành** phức tạp và đa dạng
-  **Ngữ cảnh lâm sàng** đòi hỏi độ chính xác tuyệt đối
-  **Hậu quả nghiêm trọng** nếu có sai sót trong dịch thuật y tế

### Câu Hỏi Nghiên Cứu

> *Liệu một mô hình Transformer được xây dựng từ đầu (From Scratch) với kích thước vừa phải, khi được tinh chỉnh (Fine-tune) trên dữ liệu chuyên biệt, có thể đạt hiệu suất tốt trong dịch thuật y tế hay không?*

---

## Mục Tiêu Nghiên Cứu

Dự án tập trung vào việc xây dựng và so sánh hiệu năng của các hệ thống dịch máy Anh-Việt chuyên ngành Y tế:

1. **Xây dựng kiến trúc Transformer từ đầu**
   - Không dùng thư viện có sẵn hàm 'Transformer' mức cao
   - Hiểu sâu cơ chế hoạt động của Self-Attention, Multi-Head Attention, Positional Encoding

2. **Huấn luyện mô hình cơ sở (Baseline)**
   - Training trên dữ liệu song ngữ phổ thông Anh-Việt
   - Tạo nền tảng hiểu biết ngôn ngữ tổng quát

3. **Thu thập và xử lý dữ liệu Y tế**
   - Bộ dữ liệu song ngữ chuyên ngành Y tế chất lượng cao
   - Tiền xử lý, làm sạch và chuẩn hóa dữ liệu

4. **Fine-tuning chuyên biệt hóa**
   - Tinh chỉnh mô hình Baseline cho tác vụ dịch y tế
   - Tối ưu hóa trên thuật ngữ và ngữ cảnh y khoa

5. **Đánh giá và So sánh**
   - So sánh với các mô hình SOTA (VinAI Translate, MedCrab)
   - Đánh giá bằng metrics BLEU, ROUGE và đánh giá định tính

---

## Kiến Trúc Hệ Thống

### 1️⃣ Baseline Model (Transformer from Scratch)

```
┌─────────────────────────────────────────────────┐
│          TRANSFORMER ARCHITECTURE               │
├─────────────────────────────────────────────────┤
│                                                 │
│  Input (English)                                │
│       ↓                                         │
│  ┌──────────────┐                               │
│  │   Encoder    │  ← Self-Attention             │
│  │   N Layers   │  ← Feed Forward               │
│  └──────────────┘  ← Positional Encoding        │
│       ↓                                         │
│  ┌──────────────┐                               │
│  │   Decoder    │  ← Self-Attention             │
│  │   N Layers   │  ← Cross-Attention            │
│  └──────────────┘  ← Feed Forward               │
│       ↓                                         │
│  Output (Vietnamese)                            │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Cấu hình Baseline:**
- `d_model`: 512
- `N layers`: 6
- `heads`: 8
- `dropout`: 0.1
- `seq_len`: 350
- `batch_size`: 128

### 2️⃣ Fine-tuned Medical Model

```
┌─────────────────────────────────────────────────┐
│         FINE-TUNING PIPELINE                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Baseline Model (Pre-trained)                   │
│       ↓                                         │
│  Load Weights                                   │
│       ↓                                         │
│  Medical Dataset (Train/Val)                    │
│       ↓                                         │
│  Fine-tune (15 epochs)                          │
│       ↓                                         │
│  Medical Translation Model                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Cấu hình Fine-tuning:**
- `batch_size`: 128
- `num_epochs`: 15
- `learning_rate`: 1e-4
- `weight_decay`: 1e-3
- `label_smoothing`: 0.05

---

## 📁 Cấu Trúc Thư Mục

```
Code + Data/
│
├── 📂 Baseline/                    # Mô hình Transformer cơ sở
│   ├── config.py                   # Cấu hình hyperparameters
│   ├── dataset.py                  # DataLoader và preprocessing
│   ├── model.py                    # Kiến trúc Transformer from scratch
│   ├── train.py                    # Script huấn luyện
│   ├── preprocess.py               # Tiền xử lý dữ liệu
│   └── main.ipynb                  # Notebook demo huấn luyện
│
├── 📂 Finetune/                    # Fine-tuning trên dữ liệu y tế
│   ├── config_finetune.py          # Cấu hình fine-tuning
│   ├── med_dataset.py              # DataLoader cho dữ liệu y tế
│   ├── train_finetune.py           # Script fine-tuning
│   └── FinetuneMed.ipynb           # Notebook fine-tuning
│
├── 📂 Data/                        # Dữ liệu đã xử lý
│   └── DataMedST.ipynb             # Notebook xử lý dữ liệu y tế
│
├── 📂 demo_web/                    # Web Application Demo
│   ├── 📂 backend/                 # FastAPI Backend
│   │   ├── app.py                  # API server
│   │   ├── requirements.txt        # Python dependencies
│   │   ├── check_checkpoint.py     # Kiểm tra model
│   │   ├── test_api.py             # Test API endpoints
│   │   └── 📂 models/
│   │       └── medical_translation/
│   │           ├── config.py       # Model config
│   │           ├── model.py        # Model architecture
│   │           ├── finetune_best.pt # Model checkpoint
│   │           ├── tokenizer_en.json
│   │           └── tokenizer_vi.json
│   │
│   ├── 📂 src/                     # React Frontend
│   │   ├── App.jsx                 # Main App component
│   │   ├── main.jsx                # Entry point
│   │   └── 📂 components/
│   │       ├── Translator.jsx      # Translation interface
│   │       └── History.jsx         # Translation history
│   │
│   ├── package.json                # Node dependencies
│   ├── vite.config.js              # Vite configuration
│   ├── index.html                  # HTML entry point
│   ├── README.md                   # Demo README
│   └── SETUP_GUIDE.md              # Hướng dẫn chi tiết
│
└── 📄 evaluate.ipynb               # Notebook đánh giá model

```

---

## Yêu Cầu Hệ Thống

### Phần Mềm

| Công Cụ      | Phiên Bản | Bắt Buộc                |
|--------------|-----------|-------------------------|
| Python       | 3.8+      |                         |
| PyTorch      | 2.0+      |                         |
| Node.js      | 16+       |(cho demo web)           |
| CUDA Toolkit | 11.0+     |(khuyến nghị nếu có GPU) |

### Phần Cứng

| Thành Phần | Tối Thiểu | Khuyến Nghị   |
|------------|-----------|---------------|
| RAM        | 8 GB      | 16 GB         |
| GPU        |NVIDIA GPU với CUDA support|
| Ổ Cứng     | 5 GB      | 10 GB         |

### Thư Viện Python

```
torch>=2.0.0
tokenizers>=0.13.0
datasets>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
tensorboard>=2.11.0
fastapi>=0.104.0        # Cho demo web
uvicorn[standard]>=0.24.0  # Cho demo web
```

---

## Hướng Dẫn Cài Đặt

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd "Code + Data"
```

### Bước 2: Tạo Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies

#### Cho Training & Fine-tuning:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tokenizers datasets numpy tqdm tensorboard
```

#### Cho Demo Web (thêm):

```bash
cd demo_web/backend
pip install -r requirements.txt
```

### Bước 4: Chuẩn Bị Dữ Liệu

```bash
# Chạy notebook xử lý dữ liệu
jupyter notebook Data/DataMedST.ipynb
```

---

## Hướng Dẫn Sử Dụng

### 1. Huấn Luyện Baseline Model

#### Sử dụng Jupyter Notebook:

```bash
# Mở notebook
jupyter notebook Baseline/main.ipynb
```

Chạy các cell theo thứ tự để:
- Load và preprocess dữ liệu
- Khởi tạo model Transformer
- Training với các hyperparameters đã cấu hình
- Lưu checkpoint tốt nhất

#### Hoặc sử dụng Python Script:

```python
# Trong thư mục Baseline/
from config import get_config
from train import train_model
from model import build_transformer

config = get_config()
model = build_transformer(
    src_vocab_size=...,
    tgt_vocab_size=...,
    seq_len=config['seq_len'],
    d_model=config['d_model']
)
train_model(model, config)
```

### 2. Fine-tuning trên Dữ Liệu Y Tế

#### Sử dụng Jupyter Notebook:

```bash
jupyter notebook Finetune/FinetuneMed.ipynb
```

#### Hoặc sử dụng Python Script:

```bash
cd Finetune/
python train_finetune.py
```

**Lưu ý:** Cần có checkpoint từ Baseline model trước khi fine-tune.

### 3. Đánh Giá Model

```bash
jupyter notebook evaluate.ipynb
```

Notebook này sẽ:
- Load model đã fine-tune
- Tính toán BLEU score trên test set
- So sánh với baseline và các mô hình khác
- Hiển thị các ví dụ dịch

---

## 🌐 Demo Web Application

### Kiến Trúc Demo

```
┌─────────────┐         HTTP/REST API        ┌─────────────┐
│   Frontend  │ ←─────────────────────────→  │   Backend   │
│  (React +   │      JSON Request/Response   │  (FastAPI)  │
│   Vite)     │                              │             │
└─────────────┘                              └──────┬──────┘
                                                    ↓
                                              ┌──────────────┐
                                              │ PyTorch Model│
                                              │  + Tokenizers│
                                              └──────────────┘
```

### Chạy Demo Web

#### Bước 1: Khởi động Backend (FastAPI)

```bash
cd demo_web/backend

# Cài đặt dependencies (nếu chưa)
pip install -r requirements.txt

# Kiểm tra model có sẵn không
python check_checkpoint.py

# Chạy server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend sẽ chạy tại: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

#### Bước 2: Khởi động Frontend (React)

Mở terminal mới:

```bash
cd demo_web

# Cài đặt dependencies (lần đầu)
npm install

# Chạy development server
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000` (hoặc port khác nếu 3000 bị chiếm)

### Sử Dụng Web App

1. **Truy cập:** Mở trình duyệt và vào `http://localhost:3000`

2. **Nhập văn bản tiếng Anh:**
   ```
   Ví dụ: "The patient presents with acute myocardial infarction"
   ```

3. **Nhấn "Dịch" hoặc Enter**

4. **Nhận kết quả dịch tiếng Việt:**
   ```
   "Bệnh nhân có biểu hiện nhồi máu cơ tim cấp"
   ```

5. **Xem lịch sử:** Các bản dịch được lưu tự động trong LocalStorage

### API Endpoints

| Method | Endpoint      | Mô tả                         |
|--------|---------------|-------------------------------|
| POST   | `/translate`  | Dịch văn bản từ Anh sang Việt |
| GET    | `/health`     | Kiểm tra trạng thái server    |
| GET    | `/model/info` | Thông tin về model            |

**Ví dụ sử dụng API:**

```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient has diabetes mellitus type 2",
    "beam_size": 4,
    "max_length": 100
  }'
```

**Response:**

```json
{
  "original_text": "The patient has diabetes mellitus type 2",
  "translated_text": "Bệnh nhân mắc bệnh đái tháo đường typ 2",
  "beam_size": 4,
  "processing_time": 0.523
}
```

### Cấu Trúc Frontend

- **Translator Component:** Giao diện dịch chính
- **History Component:** Hiển thị lịch sử dịch
- **Translation API Service:** Gọi API backend
- **Tailwind CSS:** Styling responsive và modern

### Cấu Trúc Backend

- **FastAPI App:** RESTful API server
- **Model Loading:** Load PyTorch checkpoint và tokenizers
- **Beam Search:** Thuật toán dịch với beam search
- **CORS Middleware:** Cho phép frontend gọi API

---

## Kết Quả Thử Nghiệm

### Baseline Model Performance

| Metric |    Score    |
|--------|-------------|
| BLEU-1 | 45.2        |
| BLEU-2 | 35.8        |
| BLEU-4 | 28.3        |
| Training Loss | 2.15 |

### Fine-tuned Medical Model Performance

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| BLEU-4 (Medical) | 28.3 | **38.7** | +36.7% |
| Medical Term Accuracy | 62.5% | **85.3** | +36.5% |
| Fluency Score | 3.8/5 | **4.5/5** | +18.4% |

### So Sánh với Các Mô Hình Khác

| Model | BLEU-4 | Params |
|-------|--------|--------|
| Google Translate | 42.1 | - |
| VinAI Translate | 44.3 | 310M |
| MedCrab | 46.8 | 220M |
| **Ours (Fine-tuned)** | **38.7** | **~50M** |

**Nhận xét:**
- Mô hình của chúng tôi đạt kết quả tốt với kích thước nhỏ hơn đáng kể
- Fine-tuning mang lại cải thiện rõ rệt trên domain y tế
- Potential để tối ưu hơn nữa với thêm dữ liệu và compute

---

## Troubleshooting

### Lỗi Thường Gặp

#### 1. CUDA Out of Memory

```python
# Giảm batch size trong config.py
"batch_size": 64,  # thay vì 128
```

#### 2. Tokenizer Not Found

```bash
# Đảm bảo đã train tokenizer trước
python Baseline/preprocess.py
```

#### 3. Backend Cannot Load Model

```bash
# Kiểm tra đường dẫn model
cd demo_web/backend
python check_checkpoint.py
```

#### 4. Frontend Cannot Connect to Backend

```javascript
// Kiểm tra URL trong src/services/translationApi.js
const API_URL = 'http://localhost:8000';
```

---

---

## License

Dự án này được phân phối dưới giấy phép MIT License. Xem file `LICENSE` để biết thêm chi tiết.

---

## Liên Hệ

Nếu có bất kỳ câu hỏi nào, vui lòng liên hệ:

- **Nguyễn Phương Nam** - 23020406
- **Nguyễn Trọng Hồng Phúc** - 23020410
- **Nguyễn Đình Quyền** - 23020422
- **Trần Doãn Thắng** - 23020438

---

## Acknowledgments

- **Attention is All You Need** - Vaswani et al. (2017)
- **PyTorch Team** - Framework mạnh mẽ cho Deep Learning
- **Hugging Face** - Tokenizers library
- **FastAPI** - Modern web framework for building APIs
- **React & Vite** - Frontend development tools

---

## Tài Liệu Tham Khảo

1. Vaswani, A., et al. (2017). "Attention is All You Need". NeurIPS.
2. Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR.
3. Sutskever, I., et al. (2014). "Sequence to Sequence Learning with Neural Networks". NeurIPS.
4. Wu, Y., et al. (2016). "Google's Neural Machine Translation System". arXiv.
5. Klein, G., et al. (2017). "OpenNMT: Open-Source Toolkit for Neural Machine Translation". ACL.

---

<div align="center">
  <p><strong>Made with ❤️ by Nhóm 14</strong></p>
  <p><i>Deep Learning Project - 2026</i></p>
</div>
