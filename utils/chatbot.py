from groq import Groq
import streamlit as st

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")

    return Groq(api_key=api_key)


def get_system_prompt(lang="EN"):
    if lang == "VI":
        return """Bạn là chuyên gia tư vấn bất động sản tại Ames, Iowa (Mỹ).
Nhiệm vụ: Tư vấn chân thực, chi tiết dựa trên kết quả định giá AI.
Phong cách: Thân thiện, chuyên nghiệp, đưa ra ý kiến rõ ràng (nên mua / không nên mua + lý do cụ thể)."""
    else:
        return """You are an expert real estate advisor in Ames, Iowa with 15+ years of experience.
Give honest, detailed, and balanced advice based on the AI valuation.
When asked "Should I buy?", give clear recommendation with specific reasons about price, quality, location, and risks."""


def get_context_message(r: dict, lang="EN") -> str:
    """Context cho Ames Housing"""
    predicted = r.get("predicted", 0)
    low = r.get("low", 0)
    high = r.get("high", 0)

    if lang == "VI":
        return f"""Thông tin nhà vừa được AI định giá:
- Khu vực: {r.get('neighborhood', 'N/A')}
- Loại nhà: {r.get('building_type', 'N/A')} | Kiểu: {r.get('house_style', 'N/A')}
- Diện tích: {r.get('area_sqft', 0):,} sq ft
- Phòng ngủ: {r.get('bedrooms', 0)} | Phòng tắm: {r.get('bathrooms', 0)}
- Năm xây: {r.get('year_built', 0)} ( {r.get('property_age', 0)} tuổi )
- Chất lượng: {r.get('overall_quality', 0)}/10
- Có Garage: {'Có' if r.get('has_garage') else 'Không'}
- Có Basement: {'Có' if r.get('has_basement') else 'Không'}
- Giá AI ước tính: ${predicted:,.0f}
- Khoảng giá: ${low:,.0f} - ${high:,.0f}
- Giá/sqft: ${r.get('price_per_sqft', 0):.1f}

Hãy tư vấn dựa trên thông tin này."""
    else:  # English
        return f"""Property just valuated by AI (Ames Housing):
- Neighborhood: {r.get('neighborhood', 'N/A')}
- Building Type: {r.get('building_type', 'N/A')} | Style: {r.get('house_style', 'N/A')}
- Living Area: {r.get('area_sqft', 0):,} sq ft
- Bedrooms: {r.get('bedrooms', 0)} | Bathrooms: {r.get('bathrooms', 0)}
- Year Built: {r.get('year_built', 0)} (Age: {r.get('property_age', 0)} years)
- Overall Quality: {r.get('overall_quality', 0)}/10
- Has Garage: {'Yes' if r.get('has_garage') else 'No'}
- Has Basement: {'Yes' if r.get('has_basement') else 'No'}
- Has Fireplace: {'Yes' if r.get('has_fireplace') else 'No'}
- Central Air: {'Yes' if r.get('has_central_air') else 'No'}
- AI Estimated Price: ${predicted:,.0f}
- Price Range: ${low:,.0f} - ${high:,.0f}
- Price per sq ft: ${r.get('price_per_sqft', 0):.1f}/sqft"""


def chat_with_advisor(messages: list, lang="EN") -> str:
    try:
        client = get_groq_client()
        
        property_context = ""
        for msg in messages:
            if isinstance(msg.get("content"), str) and "AI Estimated Price" in msg.get("content", ""):
                property_context = msg["content"]
                break

        system_message = get_system_prompt(lang)
        
        if property_context:
            full_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": property_context},
                {"role": "assistant", "content": "I understand the property details. How can I help you with this valuation?"}
            ] + [msg for msg in messages if msg.get("role") in ["user", "assistant"]][-4:]
        else:
            full_messages = messages

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # ← Model mới
            messages=full_messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Groq Error: {e}")
        return "❌ Sorry, I'm having trouble connecting to Groq right now. Please try again in a few seconds."