import React from 'react';

/**
 * Component History - Lịch sử dịch
 * 
 * Chức năng:
 * - Hiển thị 10 câu dịch gần nhất
 * - Câu mới nhất ở trên cùng
 * - Click vào mục lịch sử để nạp lại vào khu vực dịch
 * 
 * Props:
 * @param {Array} history - Mảng các object {id, sourceText, translatedText, timestamp}
 * @param {function} onHistoryItemClick - Callback khi click vào mục lịch sử
 */
const History = ({ history, onHistoryItemClick, onClearHistory }) => {
  // Format thời gian hiển thị
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / 1000 / 60);

    if (diffInMinutes < 1) return 'Vừa xong';
    if (diffInMinutes < 60) return `${diffInMinutes} phút trước`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)} giờ trước`;
    return date.toLocaleDateString('vi-VN');
  };

  // Rút gọn text nếu quá dài
  const truncateText = (text, maxLength = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Lịch Sử Dịch</h2>
          <p className="text-gray-600 text-sm mt-1">
            {history.length > 0 
              ? `${history.length} bản dịch gần nhất` 
              : 'Chưa có lịch sử dịch'}
          </p>
        </div>
        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors duration-200"
            title="Xóa tất cả lịch sử"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Xóa lịch sử
          </button>
        )}
      </div>

      {/* Danh sách lịch sử */}
      {history.length === 0 ? (
        // Empty state
        <div className="text-center py-12">
          <svg 
            className="mx-auto h-16 w-16 text-gray-400 mb-4" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" 
            />
          </svg>
          <p className="text-gray-500 text-lg">Chưa có lịch sử dịch</p>
          <p className="text-gray-400 text-sm mt-2">
            Các bản dịch của bạn sẽ được lưu lại ở đây
          </p>
        </div>
      ) : (
        // History list
        <div className="space-y-4 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
          {history.map((item) => (
            <div
              key={item.id}
              onClick={() => onHistoryItemClick(item)}
              className="history-item"
            >
              {/* Timestamp */}
              <div className="text-xs text-gray-500 mb-2">
                {formatTime(item.timestamp)}
              </div>

              {/* Source text (English) */}
              <div className="mb-3">
                <div className="flex items-center mb-1">
                  <span className="text-xs font-medium text-gray-600 bg-gray-100 px-2 py-0.5 rounded">
                    EN
                  </span>
                </div>
                <p className="text-gray-800 text-sm leading-relaxed">
                  {truncateText(item.sourceText)}
                </p>
              </div>

              {/* Divider */}
              <div className="border-t border-gray-200 my-2"></div>

              {/* Translated text (Vietnamese) */}
              <div>
                <div className="flex items-center mb-1">
                  <span className="text-xs font-medium text-medical-blue-700 bg-medical-blue-100 px-2 py-0.5 rounded">
                    VI
                  </span>
                </div>
                <p className="text-gray-800 text-sm leading-relaxed">
                  {truncateText(item.translatedText)}
                </p>
              </div>

              {/* Click hint */}
              <div className="mt-3 text-xs text-gray-400 flex items-center">
                <svg 
                  className="w-3 h-3 mr-1" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={2} 
                    d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" 
                  />
                </svg>
                Click để tải lại
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default History;
