import { useState, useEffect } from 'react';
import Translator from './components/Translator';
import History from './components/History';
import { translateMedicalText } from './services/translationApi';

/**
 * App Component - Component chính của ứng dụng
 * 
 * Quản lý:
 * - State của văn bản nguồn và kết quả dịch
 * - State loading khi đang dịch
 * - Lịch sử dịch (tối đa 10 mục, lưu trong localStorage)
 * - Logic dịch và xử lý lỗi
 */
function App() {
  // State quản lý văn bản
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // State quản lý lịch sử dịch
  const [history, setHistory] = useState([]);

  // Load lịch sử từ localStorage khi khởi động
  useEffect(() => {
    const savedHistory = localStorage.getItem('translationHistory');
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory);
        setHistory(parsedHistory);
      } catch (error) {
        console.error('Error loading history from localStorage:', error);
      }
    }
  }, []);

  // Lưu lịch sử vào localStorage mỗi khi thay đổi
  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem('translationHistory', JSON.stringify(history));
    }
  }, [history]);

  /**
   * Xử lý dịch văn bản
   */
  const handleTranslate = async () => {
    // Validate input
    if (!sourceText.trim()) {
      alert('Vui lòng nhập văn bản cần dịch');
      return;
    }

    setIsLoading(true);
    setTranslatedText(''); // Clear kết quả cũ

    try {
      // Gọi API dịch (hiện tại là giả lập)
      const result = await translateMedicalText(sourceText);
      
      // Cập nhật kết quả
      setTranslatedText(result);

      // Thêm vào lịch sử
      addToHistory(sourceText, result);

    } catch (error) {
      console.error('Translation error:', error);
      alert(error.message || 'Có lỗi xảy ra khi dịch văn bản. Vui lòng thử lại.');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Thêm bản dịch vào lịch sử
   * Giữ tối đa 10 mục, mục mới nhất ở đầu
   */
  const addToHistory = (source, translated) => {
    const newItem = {
      id: Date.now(), // Unique ID dựa trên timestamp
      sourceText: source,
      translatedText: translated,
      timestamp: Date.now()
    };

    setHistory(prevHistory => {
      // Thêm mục mới vào đầu mảng
      const updatedHistory = [newItem, ...prevHistory];
      
      // Giới hạn 10 mục
      return updatedHistory.slice(0, 10);
    });
  };

  /**
   * Xử lý khi click vào mục lịch sử
   * Nạp lại nội dung vào khu vực dịch
   */
  const handleHistoryItemClick = (item) => {
    setSourceText(item.sourceText);
    setTranslatedText(item.translatedText);
    
    // Scroll lên đầu trang
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  /**
   * Xóa toàn bộ lịch sử
   */
  const handleClearHistory = () => {
    if (window.confirm('Bạn có chắc muốn xóa toàn bộ lịch sử dịch?')) {
      setHistory([]);
      localStorage.removeItem('translationHistory');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="flex items-center justify-center w-10 h-10 bg-medical-blue-600 rounded-lg mr-3">
                <svg 
                  className="w-6 h-6 text-white" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={2} 
                    d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" 
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Medical Translation AI
                </h1>
                <p className="text-xs text-gray-500">
                  English → Vietnamese Medical Text Translation
                </p>
              </div>
            </div>
            
            <div className="hidden sm:flex items-center space-x-2 text-xs">
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full font-medium">
                ● Online
              </span>
              <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full">
                AI-Powered
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Translator Component */}
        <Translator
          sourceText={sourceText}
          translatedText={translatedText}
          isLoading={isLoading}
          onSourceTextChange={setSourceText}
          onTranslate={handleTranslate}
        />

        {/* History Component */}
        <History
          history={history}
          onHistoryItemClick={handleHistoryItemClick}
          onClearHistory={handleClearHistory}
        />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p className="mb-2">
              <strong>Medical Translation AI</strong> - Hệ thống dịch chuyên ngành y tế
            </p>
            <p className="text-xs text-gray-500">
              Được phát triển với React + Vite | Sẵn sàng tích hợp AI model thực tế
            </p>
            <p className="text-xs text-gray-400 mt-2">
              © 2025 - Dự án demo cho assignment AI dịch y tế
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
