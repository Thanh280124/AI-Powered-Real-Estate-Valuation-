from groq import Groq
import streamlit as st

def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_system_prompt(lang="VI"):
    prompts = {
        "VI": """Bạn là chuyên gia tư vấn bất động sản Việt Nam của T-Bank.
Nhiệm vụ: Tư vấn người dùng dựa trên kết quả định giá AI.
Phong cách: Thân thiện, ngắn gọn, thực tế, đưa ra ý kiến rõ ràng.
Khi được hỏi "nên mua không" hãy đưa ra nhận xét CỤ THỂ về giá, vị trí, tiềm năng.
Chỉ tư vấn về BĐS — không trả lời chủ đề khác.
Trả lời bằng Tiếng Việt.""",

        "EN": """You are T-Bank's Vietnamese real estate advisor.
Task: Advise users based on AI valuation results.
Style: Friendly, concise, practical, give clear opinions.
When asked "should I buy" give SPECIFIC feedback on price, location and potential.
Only answer real estate questions.
Reply in English.""",

        "FI": """Olet T-Bankin vietnamilainen kiinteistöneuvoja.
Tehtävä: Neuvoa käyttäjiä AI-arvioinnin tulosten perusteella.
Tyyli: Ystävällinen, ytimekäs, anna selkeitä mielipiteitä.
Vastaa vain kiinteistökysymyksiin. Vastaa suomeksi.""",

        "SV": """Du är T-Banks vietnamesiska fastighetskonsult.
Uppgift: Ge råd baserat på AI-värderingsresultat.
Stil: Vänlig, kortfattad, ge tydliga åsikter.
Svara bara på fastighetsfrågor. Svara på svenska."""
    }
    return prompts.get(lang, prompts["VI"])


def get_context_message(r: dict, lang="VI") -> str:
    predicted = r.get("predicted", 0)
    low = r.get("low", 0)
    high = r.get("high", 0)
    area = r.get("area", 0)

    templates = {
        "VI": f"""Thông tin BĐS vừa được định giá bởi AI:
- Loại giao dịch: {r.get('mode', 'Mua bán')}
- Địa chỉ: {r.get('district', '')}, {r.get('city', '')}
- Diện tích: {area} m²
- Phòng ngủ: {r.get('bedrooms', 0)} | Phòng tắm: {r.get('bathrooms', 0)}
- Giá AI ước tính: {predicted/1000:.2f} tỷ VNĐ
- Khoảng giá: {low/1000:.2f} tỷ - {high/1000:.2f} tỷ VNĐ
- Giá/m²: {r.get('price_per_m2', 0):.1f} triệu/m²

Hãy tư vấn dựa trên thông tin này. Khi người dùng hỏi "nên mua không" hãy đưa ra nhận xét cụ thể về giá, vị trí, và tiềm năng khu vực.""",

        "EN": f"""Property just valuated by AI:
- Transaction: {r.get('mode', 'Sale')}
- Location: {r.get('district', '')}, {r.get('city', '')}
- Area: {area} m²
- Bedrooms: {r.get('bedrooms', 0)} | Bathrooms: {r.get('bathrooms', 0)}
- AI estimated price: {predicted/1000:.2f} billion VND
- Price range: {low/1000:.2f}B - {high/1000:.2f}B VND
- Price/m²: {r.get('price_per_m2', 0):.1f} million/m²

When asked "should I buy" give specific feedback on price, location and area potential.""",

        "FI": f"""Kiinteistö juuri arvioitu tekoälyllä:
- Tyyppi: {r.get('mode', 'Myynti')}
- Sijainti: {r.get('district', '')}, {r.get('city', '')}
- Pinta-ala: {area} m²
- Makuuhuoneet: {r.get('bedrooms', 0)} | Kylpyhuoneet: {r.get('bathrooms', 0)}
- Arvioitu hinta: {predicted/1000:.2f} miljardia VND
- Hintahaarukka: {low/1000:.2f}B - {high/1000:.2f}B VND
- Hinta/m²: {r.get('price_per_m2', 0):.1f} milj./m²""",

        "SV": f"""Fastighet just värderad av AI:
- Typ: {r.get('mode', 'Försäljning')}
- Plats: {r.get('district', '')}, {r.get('city', '')}
- Area: {area} m²
- Sovrum: {r.get('bedrooms', 0)} | Badrum: {r.get('bathrooms', 0)}
- Uppskattat pris: {predicted/1000:.2f} miljarder VND
- Prisintervall: {low/1000:.2f}B - {high/1000:.2f}B VND
- Pris/m²: {r.get('price_per_m2', 0):.1f} milj./m²""",
    }
    return templates.get(lang, templates["VI"])


def chat_with_advisor(messages: list, lang="VI") -> str:
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi kết nối AI: {e}"