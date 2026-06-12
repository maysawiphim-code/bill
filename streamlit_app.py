import streamlit as st
from PIL import Image
import pytesseract
import re
import pandas as pd
import io
import cv2
import numpy as np
import concurrent.futures

# Page Config
st.set_page_config(page_title="Receipt OCR Pro", layout="wide")


def preprocess_image(img_cv):
    # แปลง grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # ขยายภาพ (ช่วยมากถ้าภาพความละเอียดต่ำ)
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # ลด noise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    return gray


def clean_ocr_text(text):
    if not text:
        return ""
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines if line.strip()]
    text = '\n'.join(cleaned_lines)

    # Fuzzy mapping for CJ Express and other common OCR errors
    fuzzy_replacements = {
        "สาขา": "สาขา",
        "สาราทิ": "สาขา",
        "มอดราม": "ยอดรวม",
        "เง็นสด": "เงินสด",
        "เป็นทอน": "เงินทอน",
        "ยอดราม": "ยอดรวม",
        "รวมทิ้งสิ้น": "รวมทั้งสิ้น",
        "เงินทอม": "เงินทอน",
        "เเงินสด": "เงินสด",
        "จํานวน": "จำนวน",
        "รวมสุทธิ": "ยอดรวม",
        "Net Total": "ยอดรวม",
        "ขีเจ": "CJ",
        "มอร์": "MORE",
        "เม็นสด": "เงินสด",
        "เง็นทอน": "เงินทอน",
        "มาท": "บาท",
        "แท": "บาท",
        "แดม": "แสตมป์",
        "อเพริคาโนเย็น":"อเมริกาโน่เย็น",
        "คาปุซิโนเย็น":"คาปูชิโน่เย็น",
        "นําทินย่นําติม":"น้ำทิพย์น้ำดื่ม",
    }
    for old, new in fuzzy_replacements.items():
        text = text.replace(old, new)
    return text


def parse_price(price_str):
    if not price_str:
        return 0.0
    # Handle comma as decimal (e.g. 10,00 -> 10.00)
    if ',' in price_str and '.' not in price_str:
        parts = price_str.split(',')
        if len(parts[-1]) == 2:
            price_str = '.'.join(parts)
    clean_price = re.sub(r'[^\d.]', '', price_str)
    try:
        return float(clean_price)
    except Exception:
        return 0.0


def extract_bill_data(text):
    date_match = re.search(r'(\d{2}[-/.]\d{2}[-/.]\d{2,4})', text)

    total_keywords = ["ยอดรวม", "Total", "รวมสุทธิ", "รวมทั้งสิ้น", "ยอดเงินสุทธิ"]
    cash_keywords = ["เงินสด", "CASH", "รับเงิน", "ชำระด้วย"]
    change_keywords = ["เงินทอน", "Change", "ทอน", "CHANGE"]
    site_keywords = ["สาขา", "Branch", "Store"]
    total_amount = 0.0
    cash = 0.0
    change = 0.0
    branch = "ไม่พบ"

    lines = text.split('\n')
    # ใช้บรรทัดแรกที่ไม่ว่างเป็นชื่อร้าน/ชื่อสินค้าหลัก
    name = lines[0].strip() if lines and lines[0].strip() else "ไม่พบ"

    for i, line in enumerate(lines):
        # Branch / สาขา
        if any(k in line for k in site_keywords):
            branch_line = line.strip()
            for k in site_keywords:
                if k in branch_line:
                    branch_line = branch_line.split(k, 1)[-1].strip(": -")
                    break
            if branch_line:
                branch = branch_line

        # Total
        if any(k in line for k in total_keywords):
            match = re.search(r'(\d+[\.,]\d{2})', line)
            if match:
                total_amount = parse_price(match.group(1))
            elif i + 1 < len(lines):
                match = re.search(r'(\d+[\.,]\d{2})', lines[i + 1])
                if match:
                    total_amount = parse_price(match.group(1))

        # Cash
        if any(k in line for k in cash_keywords):
            match = re.search(r'(\d+[\.,]\d{2})', line)
            if match:
                cash = parse_price(match.group(1))
            elif i + 1 < len(lines):
                match = re.search(r'(\d+[\.,]\d{2})', lines[i + 1])
                if match:
                    cash = parse_price(match.group(1))

        # Change
        if any(k in line for k in change_keywords):
            match = re.search(r'(\d+[\.,]\d{2})', line)
            if match:
                change = parse_price(match.group(1))
            elif i + 1 < len(lines):
                match = re.search(r'(\d+[\.,]\d{2})', lines[i + 1])
                if match:
                    change = parse_price(match.group(1))

    return {
        "date": date_match.group(1) if date_match else "ไม่พบ",
        "total_amount": total_amount,
        "change": change,
        "cash": cash,
        "Branch": branch,
        "name": name,
    }


def extract_items(text):
    items = []
    skip_keywords = [
        "ยอดรวม", "เงินสด", "เงินทอน", "Total", "CASH", "Change", "ทอน",
        "Vatable", "Vat", "ITEM", "บาท", "รายการ", "Tax", "TAX", "VAT",
        "ขอบคุณ", "ยินดีต้อนรับ", "ใบกำกับภาษี", "เลขที่", "สมาชิก", "รวมทั้งสิ้น"
    ]

    lines = text.split('\n')
    price_pattern = re.compile(r'(\d+[\.,]\d{2})')

    for i, line in enumerate(lines):
        # Skip dates/times
        if re.search(r'\d{2}[-/.]\d{2}[-/.]\d{2,4}', line):
            continue
        if re.search(r'^\d{2}[:\.]\d{2}$', line):
            continue
        if any(k in line for k in skip_keywords):
            continue

        p_match = price_pattern.search(line)
        if p_match:
            name = line[:p_match.start()].strip()
            # If name is empty, try the previous line
            if not name and i > 0:
                name = lines[i - 1].strip()

            if len(name) >= 2 and not any(k in name for k in skip_keywords):
                price = parse_price(p_match.group(1))
                items.append({
                    "ชื่อสินค้า": name,
                    "จำนวน": 1,
                    "ราคาต่อหน่วย": price,
                    "ยอดรวมสินค้า": price
                })
    return items


def process_image(uploaded_file):
    img_pil = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    processed = preprocess_image(img_cv)

    # ใช้ pytesseract เพียงอย่างเดียว (รองรับภาษาไทย + อังกฤษ)
    text = pytesseract.image_to_string(processed, lang='tha+eng', config='--psm 6')

    cleaned_text = clean_ocr_text(text)
    return {
        "filename": uploaded_file.name,
        "bill_data": extract_bill_data(cleaned_text),
        "receipt_items": extract_items(cleaned_text),
        "raw_text": cleaned_text
    }


# UI
st.title("🧾 Receipt OCR Pro (CJ, 7-11, Big C)")
st.markdown("ระบบดึงข้อมูลใบเสร็จอัตโนมัติ พร้อมส่งออกไฟล์ Excel แยกรายการสินค้า")

uploaded_files = st.file_uploader("อัปโหลดใบเสร็จ", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    if st.button("เริ่มประมวลผล"):
        all_results = []
        with st.spinner("กำลังวิเคราะห์..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                all_results = list(executor.map(process_image, uploaded_files))

        if all_results:
            st.success(f"ประมวลผลเสร็จสิ้น! พบ {len(all_results)} ใบเสร็จ")
            for res in all_results:
                with st.expander(f"📄 {res['filename']}", expanded=True):
                    d = res['bill_data']
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("วันที่", d['date'])
                    c2.metric("สาขา", d['Branch'])
                    c3.metric("ชื่อสินค้า", d['name'])
                    c4.metric("ยอดรวม", f"{d['total_amount']} บาท")
                    c5.metric("เงินสด", f"{d['cash']} บาท")
                    c6.metric("เงินทอน", f"{d['change']} บาท")

                    if res['receipt_items']:
                        st.table(pd.DataFrame(res['receipt_items']))
                    st.text_area("ข้อความดิบ", res['raw_text'], height=100)

            # Excel Export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_data = []
                for r in all_results:
                    summary_data.append({
                        "ชื่อไฟล์": r['filename'],
                        "วันที่": r['bill_data']['date'],
                        "สาขา": r['bill_data']['Branch'],
                        "ชื่อสินค้า": r['bill_data']['name'],
                        "ยอดรวมสุทธิ": r['bill_data']['total_amount'],
                        "เงินสด": r['bill_data']['cash'],
                        "เงินทอน": r['bill_data']['change']
                    })
                pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='สรุปยอด')

                # Sheet 2: Item Details
                items_data = []
                for r in all_results:
                    for it in r['receipt_items']:
                        items_data.append({
                            "จากไฟล์": r['filename'],
                            "วันที่": r['bill_data']['date'],
                            **it
                        })
                if items_data:
                    pd.DataFrame(items_data).to_excel(writer, index=False, sheet_name='รายละเอียดสินค้า')

            st.download_button(
                label="📥 ดาวน์โหลด Excel (แยกรายการสินค้า)",
                data=output.getvalue(),
                file_name="receipt_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
