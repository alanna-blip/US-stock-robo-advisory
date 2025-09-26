import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import json
import requests
import time
import base64
from openai import AzureOpenAI
from newsapi import NewsApiClient

# Firebase Admin SDK & Pyrebase
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase

# --- 頁面設定 ---
st.set_page_config(page_title="美股智能投顧", layout="wide")

# --- Firebase 初始化 ---
@st.cache_resource
def initialize_firebase():
    try:
        creds_base64 = os.getenv("FIREBASE_CREDS_BASE64", st.secrets.get("firebase_credentials", {}).get("base64"))
        firebase_api_key = os.getenv("FIREBASE_API_KEY", st.secrets.get("firebase_config", {}).get("apiKey"))
        if not creds_base64 or not firebase_api_key:
            if 'firebase_error' not in st.session_state:
                st.error("缺少 Firebase 憑證設定！請檢查您的環境變數或 secrets.toml。")
                st.session_state.firebase_error = True
            return None, None
        
        creds_json = base64.b64decode(creds_base64).decode("utf-8")
        firebase_creds_dict = json.loads(creds_json)
        cred = credentials.Certificate(firebase_creds_dict)
        if not firebase_admin._apps: firebase_admin.initialize_app(cred)
        
        firebase_config = {
            "apiKey": firebase_api_key,
            "authDomain": f"{firebase_creds_dict.get('project_id')}.firebaseapp.com",
            "projectId": firebase_creds_dict.get("project_id"),
            "storageBucket": f"{firebase_creds_dict.get('project_id')}.appspot.com",
            "databaseURL": f"https://{firebase_creds_dict.get('project_id')}-default-rtdb.firebaseio.com/",
        }
        try: pyrebase.initialize_app(firebase_config)
        except ValueError: pass
        
        return firestore.client(), pyrebase.initialize_app(firebase_config).auth()
    except Exception as e:
        st.error(f"Firebase 初始化失敗: {e}")
        return None, None

db, pyrebase_auth = initialize_firebase()

# --- AI & News API 函數 ---
def get_gemini_recommendation(prompt):
    api_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY"))
    if not api_key: st.error("找不到 GEMINI_API_KEY！"); return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = { "contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.5, "maxOutputTokens": 4096} }
    max_retries, backoff_factor = 3, 1.0
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            candidates = result.get("candidates")
            if not candidates or not candidates[0].get("content") or not candidates[0]["content"].get("parts"): return None
            return candidates[0]['content']['parts'][0]['text']
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1: time.sleep(backoff_factor * (2 ** attempt))
            else: return None
    return None

def get_azure_openai_recommendation(prompt):
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", st.secrets["azure_openai"]["endpoint"]),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", st.secrets["azure_openai"]["api_key"]),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", st.secrets["azure_openai"]["api_version"])
        )
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", st.secrets["azure_openai"]["deployment_name"]),
            messages=[{"role": "system", "content": "You are a professional financial advisor."}, {"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"呼叫 Azure OpenAI API 時發生錯誤: {e}")
        return None
        
@st.cache_data(ttl=3600)
def get_financial_news_summary(_tickers):
    news_api_key = os.getenv("NEWS_API_KEY", st.secrets.get("NEWS_API_KEY"))
    if not news_api_key:
        return "警告：偵測不到 NewsAPI 金鑰，無法獲取財經新聞。"
    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        query = " OR ".join(_tickers)
        articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)['articles']
        if not articles:
            return "今天沒有您投資組合的相關重大新聞。"
        news_content = ""
        for article in articles:
            news_content += f"Title: {article['title']}\nDescription: {article['description']}\n\n"
        prompt = f"請扮演一位專業的財經分析師，用繁體中文為一位投資新手，總結以下關於他們投資組合 ({', '.join(_tickers)}) 的市場新聞。請識別潛在的正面或負面訊號，並以客觀、精簡的風格分析可能帶來的影響。請將最終總結控制在 200 字以內，並直接給出結論。\n\n新聞原文如下：\n{news_content}"
        summary = get_gemini_recommendation(prompt)
        return summary if summary else "AI 無法總結新聞，請稍後再試。"
    except Exception as e:
        return f"無法獲取財經新聞：{e}。請檢查 NewsAPI 金鑰或稍後再試。"

# --- 頁面狀態管理 ---
if 'page' not in st.session_state: st.session_state.page = '登入'
if 'user' not in st.session_state: st.session_state.user = None

# --- 核心渲染函數 ---

def render_sidebar():
    with st.sidebar:
        st.title("導覽")
        if st.session_state.user:
            user_name = st.session_state.user.get('display_name', '訪客')
            st.write(f"👋 你好, {user_name}")
            if st.button("💼 我的投資組合", use_container_width=True): st.session_state.page = '我的投資組合'; st.rerun()
            if st.button("📈 個人儀表板", use_container_width=True): st.session_state.page = '儀表板'; st.rerun()
            if st.button("🤖 產生新分析", use_container_width=True): st.session_state.page = '新分析'; st.rerun()
            if st.button("📂 查看所有歷史紀錄", use_container_width=True): st.session_state.page = '歷史紀錄'; st.rerun()
        else:
            st.info("請先登入或註冊。")
            if st.session_state.page != '登入':
                if st.button("⬅️ 返回登入頁面", use_container_width=True): st.session_state.page = '登入'; st.rerun()
        st.write("---")
        st.header("資源中心")
        if st.button("🏦 一站式開戶指南", use_container_width=True): st.session_state.page = '開戶'; st.rerun()
        if st.button("📚 投資教育中心", use_container_width=True): st.session_state.page = '教育'; st.rerun()
        if st.session_state.user:
            st.write("---")
            if st.button("登出", use_container_width=True):
                st.session_state.user = None; st.session_state.page = '登入'; st.rerun()

def page_login():
    # 首先检查Firebase状态
    if not db or not pyrebase_auth:
        st.title("歡迎使用美股智能投顧")
        st.error("Firebase 初始化失敗，無法提供登入服務。請聯繫管理員。")
        return
    
    st.title("歡迎使用美股智能投顧")
    with st.container():
        st.caption("技術核心：Google Gemini 2.5 Flash & Azure OpenAI | 資料庫：Firebase Firestore")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            # 修改图片代码，添加明确的 alt 文本
            st.markdown(
                f'<img src="https://storage.googleapis.com/gemini-prod/images/workspace_emblem_2x.png" width="400" style="max-width:100%" alt="">',
                unsafe_allow_html=True
            )
            st.markdown("#### 您的個人化 AI 投資夥伴，助您輕鬆開啟美股投資之旅。")
        with col2:
            choice = st.selectbox("選擇操作", ["登入", "註冊"])
            
            if choice == "登入":
                with st.form("login_form"):
                    email, password = st.text_input("電子郵件"), st.text_input("密碼", type="password")
                    if st.form_submit_button("登入", use_container_width=True):
                        try:
                            user = pyrebase_auth.sign_in_with_email_and_password(email, password)
                            user_doc = db.collection("users").document(user['localId']).get()
                            st.session_state.user = user_doc.to_dict() if user_doc.exists else {'email': email, 'display_name': '用戶'}
                            st.session_state.user['uid'] = user['localId']
                            st.session_state.page = '儀表板'
                            st.rerun()
                        except Exception: st.error("登入失敗，請檢查您的電子郵件或密碼。")
            else: # 註冊
                with st.form("signup_form"):
                    email, password, display_name = st.text_input("電子郵件"), st.text_input("密碼", type="password"), st.text_input("暱稱")
                    if st.form_submit_button("註冊", use_container_width=True):
                        if not all([email, password, display_name]): st.warning("請填寫所有欄位。"); return
                        try:
                            user = pyrebase_auth.create_user_with_email_and_password(email, password)
                            db.collection("users").document(user['localId']).set({"email": email, "display_name": display_name, "created_at": firestore.SERVER_TIMESTAMP})
                            st.success("註冊成功！請前往登入頁面登入。")
                        except Exception: st.error("註冊失敗，該電子郵件可能已被使用或密碼格式不符(需至少6位數)。")

def page_dashboard():
    user_name = st.session_state.user.get('display_name', '訪客')
    st.title(f"📈 {user_name} 的個人儀表板")
    st.write("---")
    user_id = st.session_state.user['uid']
    latest_rec_ref = db.collection("recommendations").where("user_id", "==", user_id).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
    latest_rec = next(latest_rec_ref, None)
    
    if latest_rec:
        rec = latest_rec.to_dict()
        with st.container(border=True):
            st.subheader("📰 今日投資組合輿情分析")
            with st.spinner("正在為您分析相關財經新聞..."):
                news_summary = get_financial_news_summary(tuple(rec['tickers']))
            st.write(news_summary)
            st.caption(f"新聞來源：NewsAPI.org | AI 總結：Google Gemini")

        with st.container(border=True):
            st.subheader("📊 您最新的 AI 投資建議")
            tw_timezone = timezone(timedelta(hours=8))
            rec_time_utc = rec['timestamp']
            if rec_time_utc.tzinfo is None: rec_time_utc = rec_time_utc.replace(tzinfo=timezone.utc)
            rec_time_tw = rec_time_utc.astimezone(tw_timezone).strftime("%Y-%m-%d %H:%M:%S")
            model_used = rec.get("model", "未知模型")
            st.info(f"- **推薦時間:** {rec_time_tw}\n- **分析模型:** {model_used}\n- **AI 推薦理由:** {rec['reason']}")
            display_portfolio_performance(rec['tickers'], rec['weights'])
    else:
        st.info("您目前沒有任何 AI 推薦紀錄。")
        if st.button("🤖 點此獲取您的第一個客製化投資組合！", use_container_width=True):
            st.session_state.page = '新分析'; st.rerun()

def page_my_portfolio():
    st.title("💼 我的投資組合即時追蹤")
    st.write("---")
    user_id = st.session_state.user['uid']
    latest_rec_ref = db.collection("recommendations").where("user_id", "==", user_id).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
    latest_rec = next(latest_rec_ref, None)
    if not latest_rec:
        st.warning("您尚未產生任何 AI 投資建議。請先前往「產生新分析」頁面。")
        return
    rec = latest_rec.to_dict()
    tickers, weights = rec['tickers'], rec['weights']
    recommendation_date = rec['timestamp'].date()
    with st.spinner("正在獲取最新市場數據..."):
        try:
            data = yf.download(tickers, start=recommendation_date, end=datetime.now(), auto_adjust=True)['Close']
            if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
            
            with st.container(border=True):
                st.subheader("即時績效總覽")
                initial_investment = 10000.0
                initial_prices = data.iloc[0]
                initial_allocation = pd.Series(weights, index=tickers) * initial_investment
                shares = initial_allocation / initial_prices
                portfolio_value = (data * shares).sum(axis=1)
                current_value = portfolio_value.iloc[-1]
                previous_day_value = portfolio_value.iloc[-2] if len(portfolio_value) > 1 else initial_investment
                today_change_value = current_value - previous_day_value
                today_change_percent = (today_change_value / previous_day_value) if previous_day_value != 0 else 0
                total_return_value = current_value - initial_investment
                total_return_percent = (total_return_value / initial_investment)
                cols = st.columns(3)
                cols[0].metric(label="目前總價值 (USD)", value=f"${current_value:,.2f}", delta=f"${today_change_value:,.2f} ({today_change_percent:.2%})", help="價值基於假設的 $10,000 初始投資計算。")
                cols[1].metric(label="總報酬率", value=f"{total_return_percent:.2%}", delta=f"${total_return_value:,.2f}")
                cols[2].metric(label="追蹤天數", value=f"{(datetime.now().date() - recommendation_date).days} 天")
            
            if len(data) < 2:
                with st.container(border=True):
                    st.subheader("價值增長曲線")
                    st.info("📈 價值增長曲線將在下一個交易日後可用。")
            else:
                with st.container(border=True):
                    st.subheader("價值增長曲線")
                    fig = px.line(x=portfolio_value.index, y=portfolio_value, title="投資組合價值增長", labels={'x': '日期', 'y': '價值 (USD)'})
                    st.plotly_chart(fig, use_container_width=True)
            with st.container(border=True):
                st.subheader("目前持股明細")
                current_prices = data.iloc[-1]
                current_allocations = shares * current_prices
                breakdown_df = pd.DataFrame({"標的": tickers, "目前價值 (USD)": current_allocations, "目前佔比": (current_allocations / current_value)}).sort_values(by="目前價值 (USD)", ascending=False)
                st.dataframe(breakdown_df.style.format({"目前價值 (USD)": "${:,.2f}", "目前佔比": "{:.2%}"}), use_container_width=True)
        except Exception as e:
            st.error(f"獲取市場數據或計算績效時發生錯誤: {e}")

def page_new_analysis():
    st.title("🤖 產生新的 AI 投資建議")
    with st.form("analysis_form"):
        st.header("📋 請更新您的資訊")
        professions = ["辦公室職員", "服務業", "製造業", "公務員", "學生", "自由工作者", "其他"]
        profession = st.selectbox("職業", professions)
        salary_ranges = ["2萬以下", "2萬-4萬", "4萬-6萬", "6萬-8萬", "8萬以上"]
        monthly_salary = st.selectbox("月薪範圍（台幣）", salary_ranges)
        debt_ranges = ["無負債", "10萬以下", "10萬-50萬", "50萬-100萬", "10萬-500萬", "500萬以上"]
        debt = st.selectbox("負債範圍（台幣）", debt_ranges)
        # <-- 修正 2: 修正年齡選單 -->
        age_ranges = ["20歲以下", "20-30歲", "30-40歲", "40-50歲", "50歲以上"]
        age_range = st.selectbox("年齡範圍", age_ranges)
        st.header("📝 風險偏好與經驗")
        risk_tolerances = ["保守型", "均衡型", "積極型"]
        risk_tolerance = st.selectbox("風險偏好", risk_tolerances)
        investment_experiences = ["無經驗", "1年以下", "1-3年", "3年以上"]
        investment_experience = st.selectbox("投資經驗", investment_experiences)
        selected_model = st.selectbox("請選擇 AI 分析模型:", ("Google Gemini 2.5 Flash", "Azure OpenAI (GPT-4o mini)"))
        submitted = st.form_submit_button("🚀 開始分析", use_container_width=True)
    if submitted:
        prompt = f"使用者資料:\n- 職業: {profession}, - 月薪範圍: {monthly_salary} (台幣), - 負債範圍: {debt} (台幣)\n- 年齡範圍: {age_range}, - 風險偏好: {risk_tolerance}, - 投資經驗: {investment_experience}\n\n請根據以上資料，為一位投資新手推薦3到5個美國市場的投資標的（股票或ETF），並嚴格按照以下格式回覆:\n[START]\n推薦理由: [繁體中文，不超過150字]\n股票代碼: [例如：VOO,AAPL,MSFT]\n投資比例: [例如：0.6,0.2,0.2]\n[END]"
        with st.spinner(f"正在使用 {selected_model} 為您分析中..."):
            response_content = get_gemini_recommendation(prompt) if selected_model == "Google Gemini 2.5 Flash" else get_azure_openai_recommendation(prompt)
        if response_content:
            st.session_state.page = '儀表板'
            try:
                content = response_content.split("[START]")[1].split("[END]")[0].strip()
                lines = content.split('\n')
                reason, tickers, weights = lines[0].replace("推薦理由: ", ""), [t.strip() for t in lines[1].replace("股票代碼: ", "").split(",")], [float(w) for w in lines[2].replace("投資比例: ", "").split(",")]
                rec_data = {"user_id": st.session_state.user['uid'], "timestamp": firestore.SERVER_TIMESTAMP, "tickers": tickers, "weights": weights, "reason": reason, "model": selected_model}
                db.collection("recommendations").add(rec_data)
                st.success("分析完成並已儲存！將為您跳轉至儀表板。")
                time.sleep(2)
            except Exception as e:
                st.error(f"儲存紀錄時失敗：{e}")
            st.rerun()

def page_history():
    st.title("📂 所有歷史推薦紀錄")
    recs_ref = db.collection("recommendations").where("user_id", "==", st.session_state.user['uid']).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    user_recs = list(recs_ref)
    if not user_recs:
        st.info("您目前沒有任何歷史推薦紀錄。")
    else:
        tw_timezone = timezone(timedelta(hours=8))
        for rec_doc in user_recs:
            rec = rec_doc.to_dict()
            with st.container(border=True):
                rec_time_utc = rec['timestamp']
                if rec_time_utc.tzinfo is None: rec_time_utc = rec_time_utc.replace(tzinfo=timezone.utc)
                rec_time_tw = rec_time_utc.astimezone(tw_timezone).strftime("%Y-%m-%d %H:%M:%S")
                model_used = rec.get("model", "未知模型")
                st.subheader(f"{rec_time_tw} 的推薦 (by {model_used})")
                st.caption(f"標的: `{', '.join(rec['tickers'])}`")
                st.info(f"**當時的推薦理由：** {rec['reason']}")
                display_portfolio_performance(rec['tickers'], rec['weights'], is_historical=True)

def page_open_account():
    # <-- 修正 1: 恢復完整內容 -->
    st.title("🏦 一站式開戶指南")
    st.markdown("""
    在台灣投資美股，最常見的方式是透過國內券商的「複委託」服務。這代表您委託台灣的券商，再去美國的券商下單。
    以下推薦幾家對新手友善、手續費有競爭力的券商，幫助您輕鬆開始。
    """)
    st.subheader("1. 永豐金證券 (SinoPac Securities)")
    st.markdown("""
    - **主要特色**:
        - **豐存股-美股**: 提供定期定額/定股功能，可以一股一股或小額買入美股，非常適合小資族。
        - **數位帳戶整合**: 與自家大戶 (DAWHO) 數位銀行帳戶整合度高，資金進出方便。
        - **手續費**: 網路下單手續費具競爭力，且常有優惠活動。
    - **適合對象**: 喜歡定期定額、小額投資的年輕族群與數位帳戶使用者。
    - **[➡️ 前往永豐金證券官網](https://www.sinotrade.com.tw/)**
    """)
    st.subheader("2. 富邦證券 (Fubon Securities)")
    st.markdown("""
    - **主要特色**:
        - **市佔率高**: 為台灣最大的券商之一，系統穩定，服務據點多。
        - **手續費優惠**: 網路下單手續費低廉，是市場上的領先者之一。
        - **一戶通**: 整合台股與複委託帳戶，資金管理方便。
    - **適合對象**: 追求低手續費、希望有實體據點可諮詢的投資人。
    - **[➡️ 前往富邦證券官網](https://www.fubon.com/securities/)**
    """)
    st.subheader("3. 國泰證券 (Cathay Securities)")
    st.markdown("""
    - **主要特色**:
        - **App 介面友善**: 國泰證券 App 操作直覺，使用者體驗佳。
        - **定期定股**: 同樣提供美股定期定股功能，方便長期投資。
        - **集團資源**: 隸屬國泰金控，可與銀行、保險等服務結合。
    - **適合對象**: 重視 App 操作體驗、國泰集團的既有客戶。
    - **[➡️ 前往國泰證券官網](https://www.cathaysec.com.tw/)**
    """)
    st.warning("**溫馨提醒**: 各家券商的手續費與優惠活動時常變動，開戶前請務必前往官方網站，確認最新的費率與開戶詳情。")

def page_education_center():
    # <-- 修正 1: 恢復完整內容 -->
    st.title("📚 投資教育中心")
    education_options = [ "ETF 是什麼？", "股票風險如何評估？", "多元化投資的重要性", "手續費與交易成本", "長期投資的優勢", "如何閱讀財務報表" ]
    selected_education = st.selectbox("選擇您想學習的主題", education_options, key="education_select")
    if selected_education == "ETF 是什麼？":
        st.markdown("""
        **ETF (Exchange-Traded Fund)，中文是「指數股票型基金」**，是一種在股票交易所買賣的基金。
        您可以把它想像成一個「**投資組合懶人包**」。基金公司先幫您買好一籃子的資產（例如數十支甚至數百支股票或債券），然後將這個籃子分成很多份，讓您可以像買賣單一股票一樣，輕鬆地買賣一小份。
        - **優點**:
            - **自動分散風險**: 買一個追蹤大盤的 ETF (如 VOO)，就等於一次投資了美國 500 家大公司，避免單一公司暴跌的風險。
            - **低成本**: 管理費用通常遠低於傳統的主動型基金，長期下來可以省下可觀的成本。
            - **高透明度**: 您隨時可以知道這個「籃子」裡到底裝了哪些股票。
        - **範例**: VOO (追蹤美國 S&P 500 指數), QQQ (追蹤納斯達克 100 指數), VT (追蹤全球市場)。
        """)
    elif selected_education == "股票風險如何評估？":
        st.markdown("""
        評估股票風險沒有單一的完美指標，但您可以從以下幾個角度來綜合判斷，當個聰明的投資人：
        - **波動性 (Volatility)**: 指股價上下起伏的劇烈程度。通常用「標準差」來衡量。波動越大的股票，風險越高，但也可能帶來更高回報。
        - **Beta (β) 值**: 衡量一支股票相對於整個市場（如 S&P 500 指數）的波動性。
        - **公司基本面**: 風險不僅僅是股價波動。公司的財務狀況、產業前景、競爭力等，都是更根本的風險來源。
        - **新手建議**: 剛開始可以從大型、穩定獲利、產業龍頭的公司或大盤 ETF 入手，風險通常較低。
        """)
    elif selected_education == "多元化投資的重要性":
        st.markdown("""
        **「不要把所有雞蛋放在同一個籃子裡。」** 這句古老的諺語，完美詮釋了多元化投資的核心精神。
        多元化是指將您的資金分配到不同類型、不同產業、不同地區的資產中，目的是**分散風險**。
        - **為什麼重要？**: 降低衝擊、平滑報酬。
        - **如何做到？**: 跨資產、跨產業、跨地區。
        - **最簡單的方式**: 買入全球市場 ETF (如 VT) 或美國大盤 ETF (如 VOO)。
        """)
    elif selected_education == "手續費與交易成本":
        st.markdown("""
        **手續費是侵蝕您獲利的隱形殺手！** 在台灣透過複委託投資美股，主要會遇到 **券商手續費** (通常有最低收費) 和 **電匯費**。
        對於小額投資人來說，「最低收費」的影響最大，因此選擇有優惠的券商或使用定期定額服務非常重要。
        """)
    elif selected_education == "長期投資的優勢":
        st.markdown("""
        股神巴菲特曾說：「如果你不打算持有一支股票十年，那連十分鐘都不要持有。」
        - **享受複利效應**: 時間是您最好的朋友，讓獲利滾雪球。
        - **穿越市場波動**: 拉長時間看，優質資產的價格趨勢通常是向上的。
        - **降低擇時風險**: 避免試圖「買在最低點、賣在最高點」的徒勞無功。
        """)
    elif selected_education == "如何閱讀財務報表":
        st.markdown("""
        財務報表是公司的「體檢報告」。新手可以從理解三大核心報表開始：
        1.  **損益表 (Income Statement)**: 看公司在一段時間內是**賺錢還是虧錢** (關鍵字: 營收、淨利)。
        2.  **資產負債表 (Balance Sheet)**: 看公司在某個時間點**有多少資產、欠了多少債** (核心公式: 資產 = 負債 + 股東權益)。
        3.  **現金流量表 (Cash Flow Statement)**: 追蹤公司**現金的流入與流出**，反映真實的營運健康狀況。
        """)

def display_portfolio_performance(tickers, weights, is_historical=False):
    with st.container(border=True):
        st.write("#### 投資組合配置")
        portfolio_df = pd.DataFrame({'投資標的': tickers, '投資比例': weights})
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.dataframe(portfolio_df.assign(投資比例=lambda df: df['投資比例'].map('{:.0%}'.format)), hide_index=True)
        with col2:
            fig_pie = px.pie(portfolio_df, values='投資比例', names='投資標的', title='投資組合佔比圖')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with st.spinner("正在獲取歷史市場數據..."):
        try:
            end_date, start_date = datetime.now(), datetime.now() - timedelta(days=5*365)
            data = yf.download(tickers + ['SPY'], start=start_date, end=end_date, auto_adjust=True)["Close"]
            if data.empty or data[tickers].isnull().all().all(): st.warning("⚠️ 找不到有效的歷史數據。"); return
            rec_data, spy_data = data[tickers].ffill(), data[['SPY']].ffill()
            with st.container(border=True):
                st.subheader(f"歷史績效回測 (回測區間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
                
                st.write("##### 價格走勢 (標準化)")
                normalized_data = rec_data / rec_data.iloc[0]
                st.line_chart(normalized_data)
                
                st.write("##### 累積報酬率")
                cumulative_returns = (1 + (rec_data.pct_change().dropna() * weights).sum(axis=1)).cumprod()
                st.area_chart(cumulative_returns)
            
            with st.container(border=True):
                total_return, annual_return = cumulative_returns.iloc[-1] - 1, (1 + cumulative_returns.iloc[-1] - 1) ** (1/5) - 1
                annual_volatility = (rec_data.pct_change().dropna() * weights).sum(axis=1).std() * np.sqrt(252)
                sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility != 0 else 0
                max_drawdown = ((cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()).min()
                spy_returns = spy_data.pct_change().dropna()['SPY']
                common_index = (rec_data.pct_change().dropna() * weights).sum(axis=1).index.intersection(spy_returns.index)
                beta = (rec_data.pct_change().dropna() * weights).sum(axis=1)[common_index].cov(spy_returns[common_index]) / spy_returns[common_index].var()
                st.subheader("📊 績效總覽")
                cols = st.columns(3)
                cols[0].metric("期間總報酬率", f"{total_return:.2%}")
                cols[1].metric("年化報酬率", f"{annual_return:.2%}")
                cols[2].metric("年化波動率", f"{annual_volatility:.2%}")
                cols = st.columns(3)
                cols[0].metric("夏普比率 (Sharpe)", f"{sharpe_ratio:.2f}")
                cols[1].metric("最大回撤 (Max Drawdown)", f"{max_drawdown:.2%}", help="從最高點到最低點的最大損失幅度。")
                cols[2].metric("Beta (β) vs S&P 500", f"{beta:.2f}", help="相對於大盤的波動性。")
            
            with st.container(border=True):
                if not is_historical:
                    with st.expander("🎲 查看未來10年投資組合風險預測 (蒙地卡羅模擬)"):
                        run_monte_carlo_simulation((rec_data.pct_change().dropna() * weights).sum(axis=1))
                else:
                    st.subheader("🎲 未來10年投資組合風險預測 (蒙地卡羅模擬)")
                    run_monte_carlo_simulation((rec_data.pct_change().dropna() * weights).sum(axis=1))
        except Exception as e:
            st.error(f"⚠️ 數據處理或圖表生成失敗: {e}")

def run_monte_carlo_simulation(portfolio_returns):
    with st.spinner("正在執行 1,000 次未來路徑模擬..."):
        n_simulations, years, initial_investment = 1000, 10, 10000
        mean_return, std_dev = portfolio_returns.mean(), portfolio_returns.std()
        simulated_returns = np.random.normal(mean_return, std_dev, (252 * years, n_simulations))
        final_values = initial_investment * (1 + pd.DataFrame(simulated_returns)).cumprod().iloc[-1]
        st.subheader("十年後投資價值分佈預測")
        st.plotly_chart(px.box(y=final_values, points="all", title=f"基於過去5年數據模擬一萬美元投資十年後的價值分佈"), use_container_width=True)
        percentiles = np.percentile(final_values, [5, 50, 95])
        median_value_str, lower_bound_str, upper_bound_str = f"${percentiles[1]:,.0f}", f"${percentiles[0]:,.0f}", f"${percentiles[2]:,.0f}"
        st.markdown(f"- **中位數價值 (50% 機率)**: 10 年後，您的 ${initial_investment:,.0f} 投資，有 50% 的機率會成長到 **{median_value_str}** 美元以上。\n- **90% 信心區間**: 我們有 90% 的信心，10 年後的投資價值會落在 **{lower_bound_str}** 美元至 **{upper_bound_str}** 美元之間。")
        st.info("**解讀**: 此模擬基於過去5年的歷史波動性與回報率，推算上千種可能的未來路徑。")

# --- 主應用程式路由 ---
load_dotenv()
render_sidebar()

if 'firebase_error' in st.session_state:
    st.error("應用程式因 Firebase 設定錯誤而無法啟動。")
elif st.session_state.page == '登入':
    page_login()
elif st.session_state.page == '儀表板':
    page_dashboard()
elif st.session_state.page == '我的投資組合':
    page_my_portfolio()
elif st.session_state.page == '新分析':
    page_new_analysis()
elif st.session_state.page == '歷史紀錄':
    page_history()
elif st.session_state.page == '開戶':
    page_open_account()
elif st.session_state.page == '教育':
    page_education_center()