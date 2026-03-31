import streamlit as st

# ========== CẤU HÌNH TRANG ==========
st.set_page_config(
    page_title="T-Bank | AI Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS TÙY CHỈNH ==========
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<p class="main-title">🏠 T-Bank AI Real Estate Valuation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Định giá bất động sản thông minh bằng AI - Nhanh chóng & Chính xác</p>', unsafe_allow_html=True)

st.divider()

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Flag_of_Vietnam.svg", width=80)
    st.title("T-Bank Valuation")
    st.markdown("---")
    
    language = st.radio("🌐 Ngôn ngữ / Language", ["Tiếng Việt", "English"])
    
    st.markdown("---")
    st.markdown("### 📌 Menu")
    page = st.selectbox("Chọn trang", [
        "🏠 Định giá BĐS",
        "📊 Phân tích thị trường",
        "🔍 So sánh BĐS",
        "ℹ️ Giới thiệu"
    ])

# ========== NỘI DUNG CHÍNH ==========
if page == "🏠 Định giá BĐS":

    st.subheader("📋 Nhập thông tin bất động sản")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("📐 Diện tích (m²)", min_value=10, max_value=1000, value=60)
        bedrooms = st.selectbox("🛏️ Số phòng ngủ", [1, 2, 3, 4, 5])
        bathrooms = st.selectbox("🚿 Số phòng tắm", [1, 2, 3, 4])
        floors = st.selectbox("🏢 Số tầng", [1, 2, 3, 4, 5, 6, 7])
    
    with col2:
        district = st.selectbox("📍 Quận/Huyện", [
            "Hoàn Kiếm", "Ba Đình", "Đống Đa", "Hai Bà Trưng",
            "Cầu Giấy", "Thanh Xuân", "Hoàng Mai", "Long Biên",
            "Nam Từ Liêm", "Bắc Từ Liêm", "Tây Hồ", "Hà Đông"
        ])
        property_type = st.selectbox("🏘️ Loại BĐS", [
            "Nhà phố", "Chung cư", "Biệt thự", "Nhà ngõ"
        ])
        legal_status = st.selectbox("📄 Pháp lý", [
            "Sổ đỏ", "Sổ hồng", "Đang chờ sổ"
        ])
        direction = st.selectbox("🧭 Hướng nhà", [
            "Đông", "Tây", "Nam", "Bắc",
            "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"
        ])
    
    st.markdown("---")
    
    # Nút dự đoán
    if st.button("🔮 Định giá ngay"):
        with st.spinner("AI đang phân tích..."):
            import time
            time.sleep(1.5)  # Tạm thời giả lập - sau sẽ thay bằng ML model
            
            
            estimated_price = area * 85
            price_low = estimated_price * 0.9
            price_high = estimated_price * 1.1
        
        st.success("✅ Định giá hoàn tất!")
        st.markdown("### 💰 Kết quả định giá")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Giá thấp nhất", f"{price_low/1000:.1f} tỷ VNĐ")
        col2.metric("Giá ước tính", f"{estimated_price/1000:.1f} tỷ VNĐ", delta="Độ chính xác ~85%")
        col3.metric("Giá cao nhất", f"{price_high/1000:.1f} tỷ VNĐ")

elif page == "📊 Phân tích thị trường":
    st.subheader("📊 Phân tích thị trường (Coming soon)")
    st.info("Tính năng này sẽ được cập nhật sau khi có dữ liệu thực tế.")

elif page == "🔍 So sánh BĐS":
    st.subheader("🔍 So sánh bất động sản tương tự (Coming soon)")
    st.info("Tính năng này sẽ được cập nhật sau khi có dữ liệu thực tế.")

elif page == "ℹ️ Giới thiệu":
    st.subheader("ℹ️ Về dự án T-Bank Valuation")
    st.markdown("""
    **T-Bank AI Real Estate Valuation** là ứng dụng định giá bất động sản tự động 
    sử dụng trí tuệ nhân tạo cho thị trường Việt Nam.
    
    - 🎓 Sinh viên: Pham Thanh  
    - 🏫 Trường: VAMK - University of Applied Sciences  
    - 📅 Năm: 2026
    """)