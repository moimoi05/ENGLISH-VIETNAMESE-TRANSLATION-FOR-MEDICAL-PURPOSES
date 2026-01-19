# Medical Translation AI - Frontend Application

Ứng dụng dịch văn bản y tế từ tiếng Anh sang tiếng Việt sử dụng AI.

## 🚀 Công Nghệ Sử Dụng

- **React 18** - Thư viện UI
- **Vite** - Build tool & dev server
- **Tailwind CSS** - Styling framework
- **LocalStorage** - Lưu trữ lịch sử dịch

## 📋 Yêu Cầu Hệ Thống

- Node.js >= 16.0.0
- npm >= 7.0.0

## 🔧 Cài Đặt & Chạy

### 1. Cài đặt dependencies

```bash
npm install
```

### 2. Chạy development server

```bash
npm run dev
```

Ứng dụng sẽ chạy tại: **http://localhost:3000**

### 3. Build production

```bash
npm run build
```

### 4. Preview production build

```bash
npm run preview
```

## 📁 Cấu Trúc Dự Án

```
demo_web/
├── src/
│   ├── components/
│   │   ├── Translator.jsx      # Component khu vực dịch chính
│   │   └── History.jsx          # Component lịch sử dịch
│   ├── services/
│   │   └── translationApi.js    # Logic gọi API (giả lập)
│   ├── App.jsx                  # Component chính
│   ├── main.jsx                 # Entry point
│   └── index.css                # Tailwind + custom styles
├── public/                      # Static assets
├── index.html                   # HTML template
├── package.json                 # Dependencies
├── vite.config.js              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
└── README.md                    # Documentation
```

## ✨ Tính Năng

### 1. Khu Vực Dịch Chính
- Textarea nhập văn bản tiếng Anh (y tế)
- Textarea hiển thị kết quả dịch tiếng Việt (read-only)
- Nút "Dịch" với trạng thái loading
- Đếm số ký tự real-time
- Responsive design

### 2. Lịch Sử Dịch
- Lưu tối đa 10 bản dịch gần nhất
- Hiển thị cả văn bản gốc và đã dịch
- Hiển thị thời gian (vừa xong, X phút trước, X giờ trước)
- Click để tải lại vào khu vực dịch
- Lưu trữ persistent trong localStorage

### 3. UI/UX
- Giao diện sạch, tối giản, chuyên nghiệp
- Phong cách medical-tech với màu xanh dương
- Responsive trên mọi kích thước màn hình
- Loading state với animation
- Custom scrollbar
- Hover effects mượt mà

## 🔌 Tích Hợp AI Model Thực Tế

Hiện tại ứng dụng sử dụng hàm giả lập trong `src/services/translationApi.js`.

### Để tích hợp AI model thật:

1. **Mở file `src/services/translationApi.js`**

2. **Thay thế hàm `translateMedicalText`:**

```javascript
export async function translateMedicalText(text) {
  const response = await fetch('http://localhost:8000/api/translate', {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      // Thêm token nếu cần:
      // 'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({ 
      text, 
      source: 'en', 
      target: 'vi' 
    })
  });
  
  if (!response.ok) {
    throw new Error('Translation failed');
  }
  
  const data = await response.json();
  return data.translatedText;
}
```

3. **Hoặc tích hợp trực tiếp với OpenAI/Claude:**

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'YOUR_API_KEY',
  dangerouslyAllowBrowser: true // Chỉ dùng cho dev
});

export async function translateMedicalText(text) {
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "You are a medical translation expert. Translate English medical text to Vietnamese accurately."
      },
      {
        role: "user",
        content: `Translate this medical text to Vietnamese: ${text}`
      }
    ]
  });
  
  return response.choices[0].message.content;
}
```

**Lưu ý:** Trong production, nên gọi API thông qua backend để bảo mật API key.

## 🎨 Customization

### Thay đổi màu chủ đạo

Mở `tailwind.config.js` và chỉnh sửa:

```javascript
theme: {
  extend: {
    colors: {
      'medical-blue': {
        // Thay đổi các giá trị này
        500: '#3b82f6',
        600: '#2563eb',
        700: '#1d4ed8',
      },
    },
  },
}
```

### Thay đổi số lượng lịch sử lưu trữ

Mở `src/App.jsx`, tìm và thay đổi:

```javascript
return updatedHistory.slice(0, 10); // Thay 10 thành số khác
```

## 🧪 Demo Data

Ứng dụng có sẵn một số thuật ngữ y tế để demo trong `translationApi.js`:
- acute respiratory distress
- myocardial infarction
- diabetes mellitus
- hypertension
- pneumonia
- và nhiều hơn nữa...

## 📝 License

MIT License - Tự do sử dụng cho mục đích học tập và thương mại.

## 👨‍💻 Phát Triển

Dự án được phát triển cho assignment AI dịch y tế. Sẵn sàng tích hợp với backend và AI model thực tế.

## 🐛 Báo Lỗi & Đóng Góp

Nếu phát hiện lỗi hoặc có đề xuất cải tiến, vui lòng tạo issue hoặc pull request.

---

**Chúc bạn phát triển thành công! 🎉**
