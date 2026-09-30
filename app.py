import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# إعدادات الصفحة للتجاوب مع شاشات الهواتف
st.set_page_config(
    page_title="حاسبة EV ومحاكي التداول",
    page_icon="📈",
    layout="wide"
)

st.title("📈 حاسبة القيمة المتوقعة (EV)")
st.caption("توقع احتمالات ضرب الـ TP والـ SL باستخدام نموذج الحركة البراونية الهندسية (GBM)")

# المدخلات في الشريط الجانبي
st.sidebar.header("⚙️ إعدادات الصفقة")
ticker_symbol = st.sidebar.text_input("رمز السهم (Ticker)", value="AAPL").upper()
tp_pct = st.sidebar.number_input("هدف الربح TP (%)", value=5.0, step=0.5) / 100.0
sl_pct = st.sidebar.number_input("وقف الخسارة SL (%)", value=3.0, step=0.5) / 100.0
capital = st.sidebar.number_input("رأس المال المخصص ($)", value=1000.0, step=100.0)
forecast_days = st.sidebar.slider("عدد أيام المحاكاة", min_value=7, max_value=90, value=30)
num_sims = st.sidebar.select_slider("عدد مسارات مونت كارلو", options=[1000, 2000, 5000], value=2000)

if st.sidebar.button("🚀 تشغيل المحاكاة", use_container_width=True):
    with st.spinner("جاري جلب البيانات التاريخية وتنفيذ المحاكاة..."):
        try:
            # 1. جلب البيانات
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            
            if df.empty:
                st.error("❌ لم يتم العثور على بيانات لرمز السهم! تأكد من كتابته بشكل صحيح.")
            else:
                S0 = df['Close'].iloc[-1]
                log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
                
                trading_days = 252
                sigma = log_returns.std() * np.sqrt(trading_days)
                mu = (log_returns.mean() * trading_days) + (0.5 * sigma**2)
                
                tp_level = S0 * (1 + tp_pct)
                sl_level = S0 * (1 - sl_pct)
                dt = 1 / trading_days
                
                # 2. إجراء محاكاة مونت كارلو
                price_paths = np.zeros((forecast_days + 1, num_sims))
                price_paths[0] = S0
                
                for t in range(1, forecast_days + 1):
                    Z = np.random.standard_normal(num_sims)
                    price_paths[t] = price_paths[t-1] * np.exp(
                        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
                    )
                
                # 3. حساب الاحتمالات والأسبقية
                hit_tp, hit_sl, neither = 0, 0, 0
                for i in range(num_sims):
                    path = price_paths[:, i]
                    tp_idx = np.where(path >= tp_level)[0]
                    sl_idx = np.where(path <= sl_level)[0]
                    
                    f_tp = tp_idx[0] if len(tp_idx) > 0 else 999
                    f_sl = sl_idx[0] if len(sl_idx) > 0 else 999
                    
                    if f_tp < f_sl:
                        hit_tp += 1
                    elif f_sl < f_tp:
                        hit_sl += 1
                    else:
                        neither += 1
                
                prob_tp = hit_tp / num_sims
                prob_sl = hit_sl / num_sims
                
                num_shares = capital / S0
                gain_dollar = (tp_level - S0) * num_shares
                loss_dollar = (S0 - sl_level) * num_shares
                ev = (prob_tp * gain_dollar) - (prob_sl * loss_dollar)
                
                # 4. عرض البطاقات الإحصائية
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                col1.metric("السعر الحالي", f"${S0:.2f}")
                col2.metric("القيمة المتوقعة (EV)", f"${ev:.2f}", 
                            delta="ميزة إيجابية (+EV)" if ev > 0 else "سلبية (-EV)", 
                            delta_color="normal" if ev > 0 else "inverse")
                col3.metric("احتمال تحقق TP", f"{prob_tp*100:.1f}%", delta=f"+${gain_dollar:.2f}")
                col4.metric("احتمال ضرب SL", f"{prob_sl*100:.1f}%", delta=f"-${loss_dollar:.2f}", delta_color="inverse")
                
                st.markdown("---")
                
                # 5. رسم بياني تفاعلي
                fig = go.Figure()
                days_array = np.arange(forecast_days + 1)
                
                for i in range(min(30, num_sims)):
                    fig.add_trace(go.Scatter(x=days_array, y=price_paths[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                
                fig.add_hline(y=tp_level, line_dash="dash", line_color="#00CC96", annotation_text=f"TP: ${tp_level:.2f}")
                fig.add_hline(y=sl_level, line_dash="dash", line_color="#FF6666", annotation_text=f"SL: ${sl_level:.2f}")
                fig.add_hline(y=S0, line_dash="dot", line_color="#636EFA", annotation_text=f"S0: ${S0:.2f}")
                
                fig.update_layout(
                    title=f"مسارات التوقع لـ {ticker_symbol}",
                    xaxis_title="الأيام",
                    yaxis_title="السعر ($)",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"حدث خطأ أثناء التنفيذ: {str(e)}")
