import streamlit as st
from PIL import Image
import pytesseract
import re
import easyocr
import pandas as pd
import io
import cv2
import numpy as np
import concurrent.futures
import json

# Page Config
st.set_page_config(page_title="High Accuracy Receipt OCR", layout="wide")

# Initialize EasyOCR reader (Cached)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['th', 'en'])

reader = load_reader()

def clean_ocr_text(text):
    if not text: return ""
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)

def extract_with_llm(raw_text, api_key):
    """
    Use OpenAI to extract data from raw text with near 100% accuracy.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
        Extract receipt data from the following raw OCR text. 
        Return ONLY a JSON object with these keys: 
        'date' (DD/MM/YYYY), 'total_amount' (float), 'cash' (float), 'change' (float), 
        'items' (list of objects with 'name', 'qty', 'unit_price', 'total').
        
        Raw Text:
        {raw_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"AI Extraction Error: {e}")
        return None

def extract_with_regex(text):
    # Date extraction
    date_match = re.search(r'(\d{2}[-/.]\d{2}[-/.]\d{2,4})', text)
    
    # Improved Total/Cash/Change Regex
    total = re.search(r'(?:ยอดรวม|Total|รวมสุทธิ|Net Total|รวมทั้งสิ้น)\D*([\d,]+\.\d{2})', text, re.I)
    cash = re.search(r'(?:เงินสด|CASH|รับเงิน|Cash)\D*([\d,]+\.\d{2})', text, re.I)
    change = re.search(r'(?:เงินทอน|Change|ทอน|CHANGE)\D*([\d,]+\.\d{2})', text, re.I)
    
    # Items extraction (Regex-based)
    items = []
    price_pattern = re.compile(r'(\d+[.,]\d{2})')
    skip = ["ยอดรวม", "เงินสด", "เงินทอน", "Total", "CASH", "Change", "Vat", "Tax", "บาท", "รายการ"]
    
    for line in text.split('\n'):
        if any(k.lower() in line.lower() for k in skip): continue
        p_match = price_pattern.search(line)
        if p_match:
            name = line[:p_match.start()].strip()
            name = re.sub(r'^\d+\s*[a-zA-Z]?\s+', '', name) # Remove qty prefix
            if len(name) > 2:
                try:
                    val = float(p_match.group(1).replace(',', '.'))
                    items.append({"name": name, "qty": 1, "unit_price": val, "total": val})
                except: pass
                
    return {
        "date": date_match.group(1) if date_match else "ไม่พบ",
        "total_amount": float(total.group(1).replace(',', '')) if total else 0.0,
        "cash": float(cash.group(1).replace(',', '')) if cash else 0.0,
        "change": float(change.group(1).replace(',', '')) if change else 0.0,
        "items": items
    }

def process_image(uploaded_file, use_ai, api_key):
    img_pil = Image.open(uploaded_file)
    # High quality resize
    if max(img_pil.size) > 2000:
        img_pil.thumbnail((2000, 2000), Image.LANCZOS)
    
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 1. Basic OCR (EasyOCR)
    ocr_results = reader.readtext(img_cv, detail=0)
    raw_text = '\n'.join(ocr_results)
    
    # 2. Fallback to Tesseract if too short
    if len(raw_text.strip()) < 30:
        raw_text = pytesseract.image_to_string(img_pil, lang='tha+eng', config='--psm 6')
    
    cleaned_text = clean_ocr_text(raw_text)
    
    # 3. Extraction Strategy
    if use_ai and api_key:
        data = extract_with_llm(cleaned_text, api_key)
        if not data: # Fallback if AI fails
            data = extract_with_regex(cleaned_text)
    else:
        data = extract_with_regex(cleaned_text)
        
    return {
        "filename": uploaded_file.name,
        "bill_data": data,
        "raw_text": cleaned_text
    }

# --- UI ---
st.title("🚀 High Accuracy Receipt OCR")
st.sidebar.header("⚙️ การตั้งค่า")
use_ai = st.sidebar.toggle("ใช้ AI ช่วยสกัดข้อมูล (แม่นยำสูงสุด)", value=False)
api_key = st.sidebar.text_input("OpenAI API Key", type="password") if use_ai else None

if use_ai:
    st.sidebar.info("💡 การใช้ AI จะช่วยแก้คำผิดและดึงข้อมูลได้แม่นยำเกือบ 100% แม้ OCR จะอ่านเพี้ยน")

uploaded_files = st.file_uploader("อัปโหลดใบเสร็จ", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    if st.button("เริ่มประมวลผล"):
        all_results = []
        with st.spinner("กำลังประมวลผลด้วยเทคโนโลยีขั้นสูง..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, f, use_ai, api_key) for f in uploaded_files]
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())
        
        if all_results:
            st.success(f"ประมวลผลเสร็จสิ้น! พบ {len(all_results)} ใบเสร็จ")
            for res in all_results:
                with st.expander(f"📄 {res['filename']}", expanded=True):
                    d = res['bill_data']
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("วันที่", d.get('date', 'ไม่พบ'))
                    c2.metric("ยอดรวม", f"{d.get('total_amount', 0.0)} บาท")
                    c3.metric("เงินสด", f"{d.get('cash', 0.0)} บาท")
                    c4.metric("เงินทอน", f"{d.get('change', 0.0)} บาท")
                    
                    if d.get('items'):
                        st.table(pd.DataFrame(d['items']))
                    
                    st.text_area("ข้อความดิบ (Raw Text)", res['raw_text'], height=100)

            # Excel Download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary = [{"ไฟล์": r['filename'], **r['bill_data']} for r in all_results]
                # Remove nested items from summary
                for s in summary: s.pop('items', None)
                pd.DataFrame(summary).to_excel(writer, index=False, sheet_name='Summary')
            
            st.download_button("📥 ดาวน์โหลด Excel", output.getvalue(), "results.xlsx")
