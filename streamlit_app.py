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


def split_receipts_image(img_cv):
    """
    แยกภาพที่มีหลายใบเสร็จวางเรียงกัน (เช่น แนวนอน บนพื้นหลังสีต่างกัน)
    ออกเป็นภาพย่อยแต่ละใบ โดยเรียงจากซ้ายไปขวา
    ถ้าตรวจไม่พบหลายใบ จะคืนภาพเดิมเป็น list 1 ภาพ
    """
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ให้พื้นที่ใบเสร็จ (สว่างกว่า) เป็นค่า 255 (foreground)
    if np.mean(gray[thresh == 255]) < np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # ปิดช่องว่างเล็กๆ ภายในใบเสร็จเดียวกัน
    kernel = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        # ต้องมีขนาดใหญ่พอ และสูงพอที่จะเป็นใบเสร็จ ไม่ใช่จุดรบกวน
        if area < img_area * 0.03:
            continue
        if bh < h * 0.2:
            continue
        boxes.append((x, y, bw, bh))

    if len(boxes) < 2:
        return [img_cv]

    boxes.sort(key=lambda b: b[0])  # เรียงจากซ้ายไปขวา

    pad = 5
    crops = []
    for x, y, bw, bh in boxes:
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + bw + pad, w)
        y1 = min(y + bh + pad, h)
        crops.append(img_cv[y0:y1, x0:x1])

    return crops


def split_text_into_receipts(text):
    """
    แยกข้อความ OCR ที่มาจากหลายใบเสร็จซ้อน/ติดกันในภาพเดียว
    โดยใช้จุดสังเกตคือบรรทัด ID: ของแต่ละใบเสร็จ (เช่น ID:E067000002A1582)
    ถ้าพบจุดแยกน้อยกว่า 2 จุด จะคืนข้อความเดิมเป็น list 1 ชุด
    """
    lines = text.split('\n')
    marker_pattern = re.compile(r'ID\s*[:.;]?\s*[A-Za-z]?\d{6,}', re.IGNORECASE)
    marker_indices = [i for i, line in enumerate(lines) if marker_pattern.search(line)]

    if len(marker_indices) < 2:
        return [text]

    # ข้อความส่วนหัว (ก่อนใบเสร็จแรก) เช่น ชื่อสาขา - แนบไปทุกใบเสร็จ
    header = '\n'.join(lines[:marker_indices[0]]).strip()

    chunks = []
    for idx, start in enumerate(marker_indices):
        end = marker_indices[idx + 1] if idx + 1 < len(marker_indices) else len(lines)
        chunk_lines = lines[start:end]
        chunk_text = '\n'.join(chunk_lines).strip()
        if header:
            chunk_text = header + '\n' + chunk_text
        chunks.append(chunk_text)

    return chunks


def process_image(uploaded_file):
    """
    ประมวลผลไฟล์ภาพ 1 ไฟล์ ซึ่งอาจมีใบเสร็จมากกว่า 1 ใบอยู่ในภาพเดียว
    คืนค่าเป็น list ของผลลัพธ์ (1 รายการ = 1 ใบเสร็จ)
    """
    img_pil = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ขั้นที่ 1: แยกใบเสร็จที่วางเรียงกันในภาพ (ตามตำแหน่งภาพ)
    crops = split_receipts_image(img_cv)

    results = []
    receipt_counter = 0

    for crop in crops:
        processed = preprocess_image(crop)
        text = pytesseract.image_to_string(processed, lang='tha+eng', config='--psm 6')
        cleaned_text = clean_ocr_text(text)

        # ขั้นที่ 2: แยกใบเสร็จที่ซ้อน/ติดกันในภาพย่อยเดียว (ตามข้อความ ID:)
        sub_texts = split_text_into_receipts(cleaned_text)

        for sub_text in sub_texts:
            receipt_counter += 1
            multi = (len(crops) > 1) or (len(sub_texts) > 1) or (receipt_counter > 1)
            filename = f"{uploaded_file.name} - บิล {receipt_counter}" if multi else uploaded_file.name

            results.append({
                "filename": filename,
                "bill_data": extract_bill_data(sub_text),
                "receipt_items": extract_items(sub_text),
                "raw_text": sub_text
            })

    return results


# UI
st.title("🧾 Receipt OCR Pro (CJ, 7-11, Big C)")
st.markdown("ระบบดึงข้อมูลใบเสร็จอัตโนมัติ พร้อมส่งออกไฟล์ Excel แยกรายการสินค้า")

uploaded_files = st.file_uploader("อัปโหลดใบเสร็จ", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    if st.button("เริ่มประมวลผล"):
        all_results = []
        with st.spinner("กำลังวิเคราะห์..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                nested_results = list(executor.map(process_image, uploaded_files))
        # แต่ละไฟล์อาจมีหลายใบเสร็จ -> รวมเป็น list เดียว
        all_results = [item for sub in nested_results for item in sub]

        if all_results:
            st.success(f"ประมวลผลเสร็จสิ้น! พบ {len(all_results)} ใบเสร็จ จาก {len(uploaded_files)} ไฟล์")
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
                sheet_name = 'สรุปยอด'

                # ตารางที่ 1: Summary
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
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, index=False, sheet_name=sheet_name, startrow=0)

                # ตารางที่ 2: รายการสินค้า (อยู่ชีทเดียวกัน เว้น 2 แถว)
                items_data = []
                for r in all_results:
                    for it in r['receipt_items']:
                        items_data.append({
                            "จากไฟล์": r['filename'],
                            "วันที่": r['bill_data']['date'],
                            **it
                        })

                if items_data:
                    df_items = pd.DataFrame(items_data)
                    start_row_items = len(df_summary) + 3  # เว้นบรรทัดว่าง 2 บรรทัด
                    worksheet = writer.sheets[sheet_name]
                    worksheet.cell(row=start_row_items, column=1, value="รายละเอียดสินค้า")
                    df_items.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_row_items)

            st.download_button(
                label="📥 ดาวน์โหลด Excel (แยกรายการสินค้า)",
                data=output.getvalue(),
                file_name="receipt_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
