# 美股 AI 智能顧問 (AI Robo-Advisor)

<p align="center">
  <img src="https://storage.googleapis.com/gemini-prod/images/workspace_emblem_2x.png" alt="Project Logo" width="200"/>
</p>

<p align="center">
  一個提供個人化美股投資建議、並能即時追蹤績效的全端 Web 應用程式。<br />
  <strong>這是一個從 0 到 1，展現產品思維、全端技術與 DevOps 實踐的個人專案。</strong>
</p>

---

## 🚀 快速導覽 (Quick Access)

- 🎥 **專案 Demo 影片** → 
- 📊 **精華投影片** → https://gamma.app/docs/AI--o16fl07oitwzi7n
- 🌐 **線上 Live Demo** → https://portfolio-simulator-814789211485.asia-east1.run.app/
  > ⚠️ 注意：服務可能因資源限制而休眠，首次載入較慢。  

---

## 🎯 專案目標 (Problem Statement)

許多投資新手在面對美股市場時，因資訊過載與缺乏個人化指導而感到卻步。  
傳統的金融服務往往缺乏彈性，無法滿足每個人的獨特需求。  

**本專案旨在打造一個 AI 驅動的投資顧問**，為使用者提供：  
- 清晰、客製化且可持續追蹤的投資策略  
- 降低進入市場的門檻  

---

## ✨ 核心功能 (Features)

- **多模型 AI 分析引擎**  
- **個人化儀表板 (Dashboard)**  
- **AI 驅動的市場輿情分析**  
- **即時績效追蹤 (Real-time Tracking)**  
- **專業風險評估 (Max Drawdown, Beta)**  
- **雲端資料庫與身份驗證 (Firebase)**  

---

## 🛠️ 技術棧 (Tech Stack)

| 類別 | 技術 | 說明 |
|------|------|------|
| 前端 | Streamlit | 使用純 Python 快速建構互動式 Web 應用 |
| AI 模型 | Google Gemini 2.5 Flash, Azure OpenAI | 串接兩個主流 LLM API |
| 新聞來源 | NewsAPI | 即時財經新聞 API |
| 金融數據 | yfinance | Yahoo Finance API |
| 資料庫 | Firebase Firestore (NoSQL) | 儲存使用者與歷史紀錄 |
| 身份驗證 | Firebase Authentication | 安全登入與註冊 |
| 雲端部署 | Docker, Google Cloud Run, Cloud Build | 容器化與 Serverless 部署 |
| 語言 | Python 3.11 | 核心開發語言 |
| 版本控制 | Git, GitHub | 協作與版本管理 |

---

## 🏗️ 系統架構圖 (System Architecture)

```mermaid
flowchart TD

    subgraph User["👤 使用者"]
        A[Web 瀏覽器<br/>(Streamlit 前端)]
    end

    subgraph App["🚀 Cloud Run App (Docker + Streamlit)"]
        B[投資組合管理模組] 
        C[AI 模型服務串接] 
        D[新聞摘要模組] 
        E[績效追蹤與金融指標]
    end

    subgraph APIs["🌐 外部 API"]
        F[Gemini API] 
        G[Azure OpenAI API] 
        H[NewsAPI] 
        I[yfinance (Yahoo Finance)]
    end

    subgraph Firebase["☁️ Firebase 後端"]
        J[(Firestore Database)]
        K[(Authentication)]
    end

    %% Connections
    A -->|HTTP 請求| B
    A -->|登入/註冊| K
    B --> C
    B --> E
    C --> F
    C --> G
    D --> H
    E --> I
    B --> J
    D --> J
    E --> J
```

🔄 使用流程圖 (User Flow)
flowchart LR

    Start([登入/註冊]) --> Dashboard[進入個人化儀表板]
    Dashboard --> Invest[獲取 AI 投資建議<br/>（Gemini / Azure OpenAI）]
    Invest --> News[每日市場新聞摘要<br/>（NewsAPI + Gemini 分析）]
    News --> Track[即時追蹤投資組合績效<br/>（股價、報酬率、風險指標）]
    Track --> Dashboard

💻 本地開發設定 (Local Development Setup)
1.複製專案
git clone https://github.com/[您的GitHub帳號]/[您的倉庫名稱].git
cd [您的倉庫名稱]

2.建立虛擬環境
python3 -m venv .venv
# 啟用虛擬環境 (macOS/Linux)
source .venv/bin/activate

3.安裝套件
pip install -r requirements.txt

4.設定環境變數
在專案根目錄建立 .env，內容如下：
# Firebase Admin SDK 憑證 (Base64 編碼)
FIREBASE_CREDS_BASE64="ey..."

# Firebase Web API 金鑰
FIREBASE_API_KEY="AIza..."

# Gemini API 金鑰
GEMINI_API_KEY="AIza..."

# Azure OpenAI 憑證
AZURE_OPENAI_ENDPOINT="https://..."
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_API_VERSION="..."
AZURE_OPENAI_DEPLOYMENT_NAME="..."

# NewsAPI 金鑰
NEWS_API_KEY="..."

5.啟動應用程式
streamlit run app.py

☁️ 雲端部署 (Deployment)
本專案採用 Cloud Native 部署策略：
容器化：使用 Dockerfile 打包 Streamlit 應用程式
雲端打包：透過 Google Cloud Build 執行 Docker 打包，免本地安裝 Docker Desktop
Serverless 部署：將映像檔部署至 Google Cloud Run
所有 API 金鑰與憑證透過 Cloud Run 環境變數注入
實現金鑰與程式碼完全分離

📜 授權 (License)
此專案僅作為學習與展示用途，非商業金融產品。
您可自由 Fork 與參考，請標註來源 🙌
