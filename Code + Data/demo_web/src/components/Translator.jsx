import React from 'react';

/**
 * Component Translator - Khu vực dịch chính
 * 
 * Chức năng:
 * - Hiển thị 2 textarea: nhập tiếng Anh và hiển thị tiếng Việt
 * - Nút "Dịch" để thực hiện dịch
 * - Hiển thị trạng thái loading khi đang dịch
 * 
 * Props:
 * @param {string} sourceText - Văn bản tiếng Anh
 * @param {string} translatedText - Văn bản tiếng Việt đã dịch
 * @param {boolean} isLoading - Trạng thái đang dịch
 * @param {function} onSourceTextChange - Callback khi thay đổi văn bản nguồn
 * @param {function} onTranslate - Callback khi click nút Dịch
 */
const Translator = ({ 
  sourceText, 
  translatedText, 
  isLoading, 
  onSourceTextChange, 
  onTranslate 
}) => {
  // Xử lý submit form (Enter để dịch)
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isLoading && sourceText.trim()) {
      onTranslate();
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Dịch Văn Bản Y Tế
        </h2>
        <p className="text-gray-600 text-sm">
          Dịch tự động từ tiếng Anh sang tiếng Việt với AI
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Ô nhập tiếng Anh */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tiếng Anh (English)
            </label>
            <textarea
              value={sourceText}
              onChange={(e) => onSourceTextChange(e.target.value)}
              onKeyDown={(e) => {
                // Press Enter (without Shift) to translate
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!isLoading && sourceText.trim()) {
                    onTranslate();
                  }
                }
              }}
              placeholder="The patient shows signs of acute respiratory distress and requires immediate medical attention. Vital signs indicate elevated heart rate and decreased oxygen saturation..."
              className="w-full h-64 p-4 border border-gray-300 rounded-lg resize-none 
                       textarea-focus custom-scrollbar text-gray-800"
              disabled={isLoading}
              style={{ textTransform: 'none' }}
              onInput={(e) => {
                // Auto capitalize first character
                if (e.target.value.length === 1) {
                  e.target.value = e.target.value.charAt(0).toUpperCase();
                  onSourceTextChange(e.target.value);
                }
              }}
            />
            <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
              <span>{sourceText.length} ký tự</span>
              <span className="text-gray-400">
                <kbd className="px-2 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">Enter</kbd> để dịch, 
                <kbd className="ml-1 px-2 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">Shift+Enter</kbd> xuống dòng
              </span>
            </div>
          </div>

          {/* Ô hiển thị tiếng Việt */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tiếng Việt (Vietnamese)
            </label>
            <textarea
              value={translatedText}
              readOnly
              placeholder="Kết quả dịch sẽ hiển thị ở đây..."
              className="w-full h-64 p-4 border border-gray-300 rounded-lg resize-none 
                       bg-gray-50 text-gray-800 custom-scrollbar"
            />
            <div className="mt-2 text-xs text-gray-500">
              {translatedText.length} ký tự
            </div>
          </div>
        </div>

        {/* Nút Dịch và trạng thái */}
        <div className="mt-6 flex items-center justify-between">
          <div className="flex items-center">
            {isLoading && (
              <div className="flex items-center text-medical-blue-600">
                <svg className="animate-spin h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="text-sm font-medium">Đang dịch...</span>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={isLoading || !sourceText.trim()}
            className="btn-primary"
          >
            {isLoading ? 'Đang xử lý...' : 'Dịch'}
          </button>
        </div>
      </form>

      {/* Thông tin hỗ trợ */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
        <p className="text-sm text-blue-800">
          <strong>💡 Lưu ý:</strong> Hệ thống này được thiết kế đặc biệt cho văn bản y tế. 
          Kết quả dịch được tối ưu hóa cho các thuật ngữ chuyên môn trong lĩnh vực y học.
        </p>
      </div>
    </div>
  );
};

export default Translator;
