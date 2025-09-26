# 使用官方推薦的 Python 基礎映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製套件需求檔案
COPY requirements.txt ./requirements.txt

# 安裝所有需要的套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式的所有檔案到工作目錄
COPY . .

# 設定 Streamlit 伺服器運行的端口
EXPOSE 8080

# 容器啟動時要執行的指令
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]
