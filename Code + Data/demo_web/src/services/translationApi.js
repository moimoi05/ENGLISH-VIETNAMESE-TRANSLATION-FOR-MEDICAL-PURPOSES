/**
 * Translation API Service
 * 
 * File này chứa logic gọi API dịch thuật.
 * Hiện tại đang giả lập phản hồi từ AI model.
 * 
 * KHI TÍCH HỢP MODEL AI THẬT:
 * - Thay thế hàm translateMedicalText bằng API call thực tế
 * - Có thể dùng fetch() hoặc axios để gọi backend
 * - Giữ nguyên interface: nhận vào text, trả về Promise với translated text
 */

/**
 * Danh sách câu dịch giả lập cho demo
 * Mapping một số thuật ngữ y tế phổ biến
 */
const mockTranslations = {
  'acute respiratory distress': 'suy hô hấp cấp tính',
  'chronic obstructive pulmonary disease': 'bệnh phổi tắc nghẽn mạn tính',
  'myocardial infarction': 'nhồi máu cơ tim',
  'hypertension': 'tăng huyết áp',
  'diabetes mellitus': 'đái tháo đường',
  'cerebrovascular accident': 'tai biến mạch máu não',
  'pneumonia': 'viêm phổi',
  'patient': 'bệnh nhân',
  'symptoms': 'triệu chứng',
  'diagnosis': 'chẩn đoán',
  'treatment': 'điều trị',
  'medication': 'thuốc',
  'prescription': 'đơn thuốc',
  'vital signs': 'dấu hiệu sinh tồn',
  'blood pressure': 'huyết áp',
  'heart rate': 'nhịp tim',
  'temperature': 'nhiệt độ',
  'respiratory rate': 'nhịp thở'
};

/**
 * Hàm dịch văn bản y tế từ tiếng Anh sang tiếng Việt
 * 
 * @param {string} text - Văn bản tiếng Anh cần dịch
 * @returns {Promise<string>} - Văn bản tiếng Việt đã dịch
 * 
 * Lưu ý: Đây là hàm giả lập. Trong thực tế, hàm này sẽ:
 * 1. Gọi API backend (ví dụ: POST /api/translate)
 * 2. Backend gọi AI model (GPT, Claude, hoặc model custom)
 * 3. Nhận kết quả và trả về
 * 
 * Ví dụ tích hợp thật:
 * 
 * export async function translateMedicalText(text) {
 *   const response = await fetch('http://localhost:8000/api/translate', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify({ text, source: 'en', target: 'vi' })
 *   });
 *   
 *   if (!response.ok) {
 *     throw new Error('Translation failed');
 *   }
 *   
 *   const data = await response.json();
 *   return data.translatedText;
 * }
 */
export async function translateMedicalText(text) {
  try {
    // Validate input
    if (!text || text.trim().length === 0) {
      throw new Error('Vui lòng nhập văn bản cần dịch');
    }

    // Gọi API backend thật
    const response = await fetch('http://localhost:8000/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
  } catch (error) {
    console.error('Translation error:', error);
    throw error;
  }
}

/**
 * Hàm kiểm tra kết nối đến API (dùng khi tích hợp thật)
 * 
 * @returns {Promise<boolean>} - true nếu kết nối thành công
 */
export async function checkApiConnection() {
  try {
    // Khi có backend thật, uncomment và sửa URL:
    // const response = await fetch('http://localhost:8000/api/health');
    // return response.ok;
    
    // Hiện tại giả lập luôn kết nối thành công
    return Promise.resolve(true);
  } catch (error) {
    console.error('API connection failed:', error);
    return false;
  }
}
