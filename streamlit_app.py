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

# Page Config
st.set_page_config(page_title="Receipt OCR Scanner", layout="wide")

# Initialize EasyOCR reader (Cached for performance)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['th', 'en'])

reader = load_reader()

def clean_ocr_text(text):
    if not text: return ""
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines if line.strip()]
    text = '\n'.join(cleaned_lines)
    replacements = {
        "น ้ ้ า ท ิ น ย": "น้ำทิพย์", 
        "ย อ ด ร ว ม": "ยอดรวม", 
        "เง ิ น ส ด": "เงินสด", 
        "เง ิ น ท อ น": "เงินทอน",
        "ท อ น": "ทอน"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_bill_data(text):
    # Date extraction
    date_match = re.search(r'(\d{2}[-/.]\d{2}[-/.]\d{2,4})', text)
    
    # Total patterns
    total_patterns = [
        r'(?:ยอดรวม|Total|รวมสุทธิ|Net Total|ยอดเงินสุทธิ|Grand Total|Total\s*\(.*?\))\D*([\d,]+\.\d{2})',
        r'(?:Total|ยอดรวม|สุทธิ|รวมทั้งสิ้น)\D*([\d,]+\.\d{2})',
        r'TOTAL\s+([\d,]+\.\d{2})'
    ]
    total_amount = 0.0
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            total_amount = float(match.group(1).replace(',', ''))
            break
            
    # Cash patterns
    cash_patterns = [
        r'(?:เงินสด|CASH|รับเงิน|ชำระด้วย|Cash)\D*([\d,]+\.\d{2})',
        r'CASH\s+([\d,]+\.\d{2})'
    ]
    cash = 0.0
    for pattern in cash_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            cash = float(match.group(1).replace(',', ''))
            break

    # Change patterns
    change_patterns = [
        r'(?:เงินทอน|Change|ทอน)\D*([\d,]+\.\d{2})',
        r'CHANGE\s+([\d,]+\.\d{2})'
    ]
    change = 0.0
    for pattern in change_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            change = float(match.group(1).replace(',', ''))
            break
            
    return {
        "date": date_match.group(1) if date_match else "ไม่พบ",
        "total_amount": total_amount,
        "change": change,
        "cash": cash,
    }

def extract_items(text):
    items = []
    skip_keywords = [
        "ยอดรวม", "เงินสด", "เงินทอน", "Total", "CASH", "Change", "ทอน",
        "Vatable", "Vat", "ITEM", "บาท", "รายการ", "Tax", "TAX", "VAT",
        "POS", "User", "ANO", "BAO", "ขอบคุณ", "ยินดีต้อนรับ", "ใบกำกับภาษี",
        "RECEIPT", "INVOICE", "ABB", "เลขที่", "เครื่อง", "พนักงาน", "สมาชิก",
        "รวมสุทธิ", "Net Total", "ยอดเงินสุทธิ", "Grand Total", "รับเงิน", "รวมทั้งสิ้น"
    ]
    
    # Improved price pattern to be more flexible with dots/commas
    price_pattern = re.compile(r'(\d+[.,]\d{2})')
    
    for line in text.split('\n'):
        line = line.strip()
        if not line or any(k.lower() in line.lower() for k in skip_keywords): 
            continue
            
        price_match = price_pattern.search(line)
        if not price_match: 
            continue
            
        before_price = line[:price_match.start()].strip()
        qty = 1
        name_part = before_price
        
        # Try to find quantity at the beginning
        qty_match = re.search(r'^(\d+)\s*[a-zA-Z]?\s+', before_price)
        if qty_match:
            try:
                qty = int(qty_match.group(1))
                name_part = before_price[qty_match.end():].strip()
            except: 
                pass
        
        # Validate name part
        if not re.search(r'[ก-๙a-zA-Z]', name_part) or len(name_part) < 2: 
            continue
            
        name_part = re.sub(r'[*&#$!]+', '', name_part).strip()
        
        try:
            # Handle both dot and comma as decimal separator
            raw_price = price_match.group(1).replace(',', '.')
            unit_price = float(raw_price)
            items.append({
                "รายการ": name_part, 
                "จำนวน": qty, 
                "ราคา/หน่วย": round(unit_price/qty, 2) if qty > 0 else unit_price, 
                "รวม": unit_price
            })
        except: 
            continue
            
    return items

def detect_and_split_image(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = thresh.shape
    v_proj = np.sum(thresh == 255, axis=0)
    center_area = range(int(w * 0.4), int(w * 0.6))
    if center_area:
        if np.max(v_proj[center_area]) > h * 0.9:
            split_x = int(w * 0.4) + np.argmax(v_proj[center_area])
            return [img_cv[:, :split_x], img_cv[:, split_x:]]
    return [img_cv]

def process_image(uploaded_file):
    img_pil = Image.open(uploaded_file)
    if max(img_pil.size) > 1500:
        img_pil.thumbnail((1500, 1500), Image.LANCZOS)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    sub_images = detect_and_split_image(img_cv)
    results = []
    for i, sub_img in enumerate(sub_images):
        try:
            # Use EasyOCR
            ocr_results = reader.readtext(sub_img, detail=0)
            text = '\n'.join(ocr_results)
            
            # Fallback to Tesseract if needed
            if len(text.strip()) < 30:
                sub_img_pil = Image.fromarray(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(sub_img_pil, lang='tha+eng', config='--psm 6')
                
            cleaned_text = clean_ocr_text(text)
            results.append({
                "filename": f"{uploaded_file.name} (ใบที่ {i+1})",
                "bill_data": extract_bill_data(cleaned_text),
                "receipt_items": extract_items(cleaned_text),
                "raw_text": cleaned_text
            })
        except Exception as e:
            st.error(f"Error processing sub-image: {e}")
    return results

# --- UI ---
st.title("🧾 Receipt OCR Scanner (Optimized)")
st.markdown("รองรับใบเสร็จ **7-11, Big C, CJ Express** และการแยกใบเสร็จอัตโนมัติ")

uploaded_files = st.file_uploader("เลือกรูปภาพใบเสร็จ", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    if st.button("เริ่มประมวลผล"):
        all_results = []
        progress_bar = st.progress(0)
        
        with st.spinner("กำลังประมวลผล OCR..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results_lists = list(executor.map(process_image, uploaded_files))
            
            for res_list in results_lists:
                all_results.extend(res_list)
        
        progress_bar.progress(100)
        
        if all_results:
            st.success(f"ประมวลผลเสร็จสิ้น! พบ {len(all_results)} ใบเสร็จ")
            
            # Display Results
            for res in all_results:
                with st.expander(f"📄 {res['filename']}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("วันที่", res['bill_data']['date'])
                    col2.metric("ยอดรวม", f"{res['bill_data']['total_amount']} บาท")
                    col3.metric("เงินสด", f"{res['bill_data']['cash']} บาท")
                    col4.metric("เงินทอน", f"{res['bill_data']['change']} บาท")
                    
                    if res['receipt_items']:
                        st.subheader("รายการสินค้า")
                        st.table(pd.DataFrame(res['receipt_items']))
                    else:
                        st.warning("ไม่พบรายการสินค้าที่ชัดเจน")
                    
                    st.subheader("ข้อความดิบจาก OCR (สำหรับตรวจสอบ)")
                    st.text_area("Raw Text", res['raw_text'], height=150)
            
            # Download Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary = []
                for r in all_results:
                    summary.append({
                        "ไฟล์": r['filename'], 
                        "วันที่": r['bill_data']['date'], 
                        "ยอดรวม": r['bill_data']['total_amount'],
                        "เงินสด": r['bill_data']['cash'],
                        "เงินทอน": r['bill_data']['change']
                    })
                pd.DataFrame(summary).to_excel(writer, index=False, sheet_name='Summary')
                
                items_all = []
                for r in all_results:
                    for it in r['receipt_items']:
                        items_all.append({"ไฟล์": r['filename'], **it})
                if items_all:
                    pd.DataFrame(items_all).to_excel(writer, index=False, sheet_name='Items')
            
            st.download_button(
                label="📥 ดาวน์โหลดผลลัพธ์เป็น Excel",
                data=output.getvalue(),
                file_name="receipt_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
