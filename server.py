import os
import re
import io
import yaml
import urllib.request
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from document_parser import extract_text_from_file
from database import (
    save_dossier, get_all_dossiers, init_db,
    save_customer_request, get_customer_requests, mark_customer_request_status
)
from exporter import generate_bid_docx
from ai_engine import extract_rfp_with_gemini, chat_with_assistant, GEMINI_API_KEY

app = FastAPI(title="Thiên Minh AI Agent - Workflow & Aura Studio", version="4.0")

from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

init_db()

def load_config():
    config_path = os.path.join(BASE_DIR, "business_rules.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

# 4 Kịch bản Sheet mẫu bám sát quy trình của Thiên Minh
SHEET_SCENARIOS = [
    {
        "id": "s1",
        "code": "MẪU 1",
        "name": "HỒ SƠ CHUẨN",
        "desc": "Đúng định dạng • Sạch • CK 4%",
        "client": "Ngân hàng TMCP Sài Gòn (SCB)",
        "deadline": "10",
        "budget": "250000000",
        "discount": "4",
        "sla": "8x5 Tiêu chuẩn NBD",
        "rows": [
            {"item": "Firewall Fortinet FortiGate 100F", "spec": "IPS >= 1Gbps, 500 VPN", "qty": "1", "mandatory": "Chính hãng CO/CQ"},
            {"item": "Switch Cisco Catalyst 1000 24-Port PoE+", "spec": "24 port PoE+, L2", "qty": "4", "mandatory": "Bảo hành 12 tháng"}
        ]
    },
    {
        "id": "s2",
        "code": "MẪU 2",
        "name": "SAI ĐỊNH DẠNG",
        "desc": "Test báo lỗi & nhập lại",
        "client": "Công ty TNHH Xây dựng Phúc Hưng",
        "deadline": "gấp trong tháng",
        "budget": "chưa có",
        "discount": "abc%",
        "sla": "",
        "rows": [
            {"item": "Switch mạng tòa nhà", "spec": "Loại tốt", "qty": "-2", "mandatory": ""},
            {"item": "", "spec": "Cục phát wifi", "qty": "nhiều", "mandatory": ""}
        ]
    },
    {
        "id": "s3",
        "code": "MẪU 3",
        "name": "DỮ LIỆU CHƯA SẠCH",
        "desc": "Test làm sạch & quay lại",
        "client": "   công ty   CP   công nghệ   tân tiến  ",
        "deadline": "12 ngày",
        "budget": "350 triệu",
        "discount": "5%",
        "sla": "8x5",
        "rows": [
            {"item": "  aruba   ap-515   wifi 6  ", "spec": "  chịu tải cao ", "qty": "8 cái", "mandatory": "lắp đặt tận nơi"},
            {"item": "thiết bị lạ chưa rõ tên xxx", "spec": "không rõ", "qty": "2", "mandatory": ""}
        ]
    },
    {
        "id": "s4",
        "code": "MẪU 4",
        "name": "VƯỢT THẨM QUYỀN",
        "desc": "CK 9% + SLA 2h • Cần duyệt",
        "client": "Tập đoàn Bán lẻ Vinamart",
        "deadline": "12",
        "budget": "700000000",
        "discount": "9",
        "sla": "24/7/2h Onsite khẩn cấp",
        "rows": [
            {"item": "Switch Cisco Catalyst 9300X 24-Port 10G Core", "spec": "24p 10G SFP+, L3", "qty": "2", "mandatory": "Stacking"},
            {"item": "Switch Cisco Catalyst 9300 48-Port PoE+", "spec": "48p PoE+, L3", "qty": "4", "mandatory": "Nguồn dự phòng"},
            {"item": "Firewall Fortinet FortiGate 100F", "spec": "Next-Gen Firewall", "qty": "1", "mandatory": "HA Ready"}
        ]
    }
]

class SheetRow(BaseModel):
    item: str = ""
    spec: str = ""
    qty: str = ""
    mandatory: str = ""
    raw_data: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

class SheetPayload(BaseModel):
    client: str = ""
    deadline: str = ""
    budget: str = ""
    discount: str = ""
    sla: str = ""
    rows: List[SheetRow] = []
    columns: Optional[List[str]] = None
    raw_rows: Optional[List[Dict[str, Any]]] = None
    extra_feedback_note: str = ""

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    context: Dict[str, Any] = {}

class SaveDossierRequest(BaseModel):
    evaluation: Dict[str, Any]
    status: str = "ISSUED"
    notes: str = "Đã qua quy trình Sheet -> Làm sạch -> AI Phân tích -> Feedback OK"

# --- BƯỚC 1: KIỂM TRA ĐỊNH DẠNG SHEET ---
@app.post("/api/validate-sheet")
async def api_validate_sheet(sheet: SheetPayload):
    errors = []
    error_fields = []
    if not sheet.client.strip():
        errors.append("Thiếu tên Khách hàng / Doanh nghiệp.")
        error_fields.append({"field": "client", "message": "Thiếu tên Khách hàng"})

    # Kiểm tra hạn chót phải có số
    if not re.search(r"\d+", sheet.deadline or ""):
        errors.append("Cột 'Thời hạn triển khai' sai định dạng (phải là số ngày, ví dụ: 10).")
        error_fields.append({"field": "deadline", "message": "Hạn triển khai sai định dạng"})

    # Kiểm tra chiết khấu nếu có nhập
    disc_raw = (sheet.discount or "0").strip().replace("%", "")
    try:
        d_val = float(disc_raw)
        if d_val < 0 or d_val > 100:
            errors.append("Cột 'Chiết khấu (%)' phải nằm trong khoảng 0 - 100.")
            error_fields.append({"field": "discount", "message": "Chiết khấu phải từ 0 - 100%"})
    except ValueError:
        errors.append(f"Cột 'Chiết khấu' sai định dạng số ('{sheet.discount}').")
        error_fields.append({"field": "discount", "message": "Chiết khấu sai định dạng"})

    # Kiểm tra ngân sách nếu có nhập
    b_raw = (sheet.budget or "").lower().replace(",", "").replace(".", "").replace("triệu", "").replace("vnđ", "").replace("đ", "").strip()
    if b_raw and not re.search(r"\d+", b_raw):
        errors.append(f"Cột 'Ngân sách' sai định dạng số ('{sheet.budget}').")
        error_fields.append({"field": "budget", "message": "Ngân sách sai định dạng"})

    if not sheet.rows:
        errors.append("Sheet phải có ít nhất 1 dòng hạng mục thiết bị/dịch vụ.")
        error_fields.append({"field": "rows", "message": "Chưa có dòng hạng mục"})
    else:
        for idx, r in enumerate(sheet.rows, 1):
            if not r.item.strip():
                errors.append(f"Dòng {idx}: Thiếu tên hạng mục thiết bị.")
                error_fields.append({"field": "row", "row_idx": idx - 1, "col": "item", "message": f"Dòng {idx}: Thiếu tên thiết bị"})
            q_raw = r.qty.lower().replace("cái", "").replace("con", "").replace("bộ", "").replace("chiếc", "").strip()
            try:
                q_num = int(q_raw)
                if q_num <= 0:
                    errors.append(f"Dòng {idx}: Số lượng phải > 0 (đang nhập '{r.qty}').")
                    error_fields.append({"field": "row", "row_idx": idx - 1, "col": "qty", "message": f"Dòng {idx}: Số lượng phải > 0"})
            except ValueError:
                errors.append(f"Dòng {idx}: Số lượng sai định dạng số ('{r.qty}').")
                error_fields.append({"field": "row", "row_idx": idx - 1, "col": "qty", "message": f"Dòng {idx}: Số lượng sai định dạng"})

    is_valid = len(errors) == 0
    format_errs = [e for e in errors if "định dạng" in e.lower() or "khoảng 0" in e.lower() or "phải > 0" in e.lower()]
    missing_errs = [e for e in errors if "thiếu" in e.lower() or "ít nhất 1 dòng" in e.lower()]
    return {
        "valid_format": is_valid,
        "errors": errors,
        "error_fields": error_fields,
        "diagnostics": {
            "format_errors": format_errs,
            "missing_fields": missing_errs,
            "duplicates": [],
            "out_of_catalog": [],
            "total_issues": len(errors),
            "has_issues": len(errors) > 0
        },
        "message": "Định dạng Sheet hợp lệ! Sẵn sàng chuyển sang bước Làm sạch dữ liệu." if is_valid else "Phát hiện lỗi định dạng Sheet! Vui lòng chỉnh sửa và nhập lại."
    }

# --- BƯỚC 2: LÀM SẠCH & CHUẨN HÓA DỮ LIỆU ---
def match_sku_from_text(text: str, catalog: list):
    t = text.lower()
    # 1. Router / Định tuyến
    if "router" in t or "định tuyến" in t or "isr" in t or "c8300" in t or "gateway" in t:
        r_cat = next((c for c in catalog if c.get("category") == "Router" or "ISR" in c.get("sku", "")), None)
        if r_cat:
            return r_cat
    # 2. Firewall / Tường lửa
    if "fortigate" in t or "100f" in t or "firewall" in t or "fg-" in t or "tường lửa" in t:
        return next((c for c in catalog if c["sku"] == "FORTI-FG-100F"), None)
    # 3. Switches Cisco
    if "9300x" in t or "quang core" in t or "10g core" in t or "25g" in t:
        return next((c for c in catalog if c["sku"] == "CISCO-C9300X-24Y"), None)
    if "9300" in t or "48-port" in t or "48 port" in t or "c9300" in t:
        return next((c for c in catalog if c["sku"] == "CISCO-C9300-48P"), None)
    if "1000" in t or "24-port" in t or "24 port" in t or "c1000" in t:
        return next((c for c in catalog if c["sku"] == "CISCO-C1000-24P"), None)
    if "switch" in t or "chuyển mạch" in t:
        c1000 = next((c for c in catalog if "CISCO-C1000" in c["sku"]), None)
        if c1000:
            return c1000
    # 4. Access Point / Wi-Fi
    if "aruba" in t or "ap-515" in t or "wifi" in t or "wi-fi" in t or "ap" in t or "phát wifi" in t:
        return next((c for c in catalog if c["sku"] == "ARUBA-AP-515"), None)
    return None

@app.post("/api/clean-sheet")
async def api_clean_sheet(sheet: SheetPayload):
    config = load_config()
    catalog = config.get("catalog", [])

    clean_client = re.sub(r"\s+", " ", sheet.client).strip().title()
    dl_m = re.search(r"(\d+)", sheet.deadline or "10")
    clean_deadline = int(dl_m.group(1)) if dl_m else 10

    disc_m = re.search(r"(\d+(?:\.\d+)?)", sheet.discount or "0")
    clean_discount = float(disc_m.group(1)) if disc_m else 0.0

    # Chuẩn hóa ngân sách
    b_str = (sheet.budget or "0").lower().replace(",", "").replace(".", "").strip()
    b_num_m = re.search(r"(\d+(?:\.\d+)?)", b_str)
    clean_budget = float(b_num_m.group(1)) if b_num_m else 0.0
    if "tỷ" in b_str or "ty" in b_str:
        clean_budget *= 1_000_000_000
    elif "triệu" in b_str or "tr" in b_str:
        clean_budget *= 1_000_000

    clean_sla = re.sub(r"\s+", " ", sheet.sla).strip() or "8x5 Tiêu chuẩn NBD"

    cleaned_rows = []
    unclean_issues = []
    unclean_error_fields = []

    diag_format_errors = []
    diag_missing = []
    diag_duplicates = []
    diag_out_of_catalog = []

    if 0 < clean_budget < 100_000:
        msg = f"Ngân sách '{sheet.budget}' chưa rõ đơn vị VNĐ (nếu là {int(clean_budget)} triệu, vui lòng nhập '{int(clean_budget)} triệu' hoặc '{int(clean_budget)*1000000}')."
        unclean_issues.append(msg)
        unclean_error_fields.append({"field": "budget", "message": "Ngân sách chưa rõ đơn vị VNĐ"})
        diag_format_errors.append({"field": "budget", "message": msg})

    # Rà soát trùng lặp mặt hàng trong cùng Sheet
    item_freq = {}
    for i, r in enumerate(sheet.rows):
        k = (r.item or "").strip().lower()
        if k:
            item_freq.setdefault(k, []).append(i)

    for k, idx_list in item_freq.items():
        if len(idx_list) > 1:
            row_nums = [f"Dòng {x + 1}" for x in idx_list]
            orig_name = sheet.rows[idx_list[0]].item
            diag_duplicates.append({
                "item": orig_name,
                "count": len(idx_list),
                "row_indices": idx_list,
                "message": f"Mặt hàng '{orig_name}' bị trùng lặp ở {', '.join(row_nums)} ({len(idx_list)} lần)."
            })

    for idx, r in enumerate(sheet.rows, 1):
        c_item = re.sub(r"\s+", " ", r.item).strip()
        c_spec = re.sub(r"\s+", " ", r.spec).strip() or "Chuẩn hãng"
        c_mand = re.sub(r"\s+", " ", r.mandatory).strip() or "Không có yêu cầu đặc biệt"
        q_m = re.search(r"(\d+)", r.qty)
        c_qty = int(q_m.group(1)) if q_m else 0

        # Kiểm tra thiếu thông tin
        if not c_item:
            diag_missing.append({
                "row_idx": idx - 1,
                "col": "item",
                "message": f"Dòng {idx}: Thiếu tên hạng mục thiết bị."
            })
        elif c_qty <= 0:
            diag_missing.append({
                "row_idx": idx - 1,
                "col": "qty",
                "message": f"Dòng {idx}: Thiếu số lượng hoặc số lượng <= 0."
            })

        # Phân biệt rõ giữa:
        # 1. "Dữ liệu rác / chưa sạch": tên rỗng, vô nghĩa (xxx), cố tình ghi "chưa rõ", "không rõ", hoặc số lượng <= 0
        # 2. "Thiết bị ngoài danh mục kho": tên hợp lệ nhưng chưa có trong kho 6 mã chuẩn của Thiên Minh
        is_junk = (
            not c_item
            or c_item.lower() in ["xxx", "thiết bị lạ chưa rõ tên xxx"]
            or "chưa rõ" in c_item.lower()
            or "chưa rõ" in c_spec.lower()
            or "không rõ" in c_item.lower()
            or c_spec.lower() == "không rõ"
            or c_qty <= 0
        )

        if is_junk:
            unclean_issues.append(f"Dòng {idx} ('{c_item or 'Trống'}'): Tên thiết bị hoặc thông số chưa rõ ràng ('{c_spec}'), cần kiểm tra lại.")
            col_target = "item"
            if "chưa rõ" in c_spec.lower() or c_spec.lower() == "không rõ":
                col_target = "spec"
            elif c_qty <= 0:
                col_target = "qty"
            unclean_error_fields.append({
                "field": "row",
                "row_idx": idx - 1,
                "col": col_target,
                "message": f"Dòng {idx}: Tên hoặc thông số chưa rõ ràng"
            })

        matched = match_sku_from_text(c_item, catalog)
        if not matched:
            diag_out_of_catalog.append({
                "row_idx": idx - 1,
                "item": c_item,
                "message": f"Dòng {idx} ('{c_item}'): Chưa thuộc danh mục kho chuẩn Thiên Minh (Cisco/Fortinet/Aruba)."
            })

        # Gợi ý thiết bị catalog thay thế phù hợp
        suggested_sku = matched["sku"] if matched else "CISCO-C1000-24P"
        suggested_name = matched["name"] if matched else "Switch Cisco Catalyst 1000 24-Port Gigabit PoE+ Layer 2"
        if not matched:
            t_low = c_item.lower()
            if "firewall" in t_low or "bảo mật" in t_low or "an ninh" in t_low:
                suggested_sku = "FORTI-FG-100F"
                suggested_name = "Firewall Fortinet FortiGate 100F Next-Gen Firewall"
            elif "wifi" in t_low or "ap" in t_low or "access point" in t_low:
                suggested_sku = "ARUBA-AP-515"
                suggested_name = "Access Point Aruba AP-515 Wi-Fi 6 (802.11ax) Doanh nghiệp"
            elif "router" in t_low or "định tuyến" in t_low or "gateway" in t_low:
                suggested_sku = "CISCO-ISR-4331"
                suggested_name = "Router Cisco ISR 4331/K9 Gigabit Security Router"
            elif "core" in t_low or "quang" in t_low or "10g" in t_low:
                suggested_sku = "CISCO-C9300X-24Y"
                suggested_name = "Switch Cisco Catalyst 9300X 24-Port 25G/10G Fiber Core"
            elif "48" in t_low or "l3" in t_low:
                suggested_sku = "CISCO-C9300-48P"
                suggested_name = "Switch Cisco Catalyst 9300 48-Port PoE+ Layer 3"

        r_raw = getattr(r, "raw_data", None) or (r.dict().get("raw_data") if hasattr(r, "dict") else None)
        cleaned_rows.append({
            "sku": matched["sku"] if matched else "OUT_OF_CATALOG",
            "raw_item": c_item,
            "clean_name": matched["name"] if matched else f"{c_item} (Ngoài danh mục kho)",
            "spec": c_spec,
            "qty": max(1, c_qty),
            "mandatory": c_mand,
            "raw_data": r_raw,
            "is_out_of_catalog": matched is None,
            "is_duplicate": len(item_freq.get(c_item.lower(), [])) > 1,
            "suggested_sku": suggested_sku,
            "suggested_name": suggested_name
        })

    is_clean = len(unclean_issues) == 0
    total_diag_issues = len(diag_format_errors) + len(diag_missing) + len(diag_duplicates) + len(diag_out_of_catalog)

    return {
        "is_clean": is_clean,
        "unclean_issues": unclean_issues,
        "error_fields": unclean_error_fields,
        "diagnostics": {
            "format_errors": diag_format_errors,
            "missing_fields": diag_missing,
            "duplicates": diag_duplicates,
            "out_of_catalog": diag_out_of_catalog,
            "total_issues": total_diag_issues,
            "has_issues": total_diag_issues > 0
        },
        "cleaned_data": {
            "client": clean_client,
            "deadline_days": clean_deadline,
            "orig_deadline_days": clean_deadline,
            "budget": clean_budget,
            "discount_pct": clean_discount,
            "orig_discount_pct": clean_discount,
            "sla": clean_sla,
            "orig_sla": clean_sla,
            "columns": sheet.columns or [],
            "raw_rows": sheet.raw_rows or [],
            "rows": cleaned_rows
        }
    }

# --- BƯỚC 3 & 4: AI AGENT PHÂN TÍCH & XÁC NHẬN FEEDBACK ---
@app.post("/api/analyze-sheet")
async def api_analyze_sheet(payload: Dict[str, Any]):
    config = load_config()
    catalog = config.get("catalog", [])
    approval_rules = config.get("approval_rules", {})
    ops = config.get("operations_capacity", {"total_capacity_mandays": 110, "booked_mandays_current_month": 68})
    rem_mandays = ops["total_capacity_mandays"] - ops["booked_mandays_current_month"]

    cleaned = payload.get("cleaned_data", payload)
    feedback_note = payload.get("extra_feedback_note", "").strip()
    retry_count = int(payload.get("retry_count", 0))
    max_retries = int(payload.get("max_retries", 2))
    auto_optimize = bool(payload.get("auto_optimize", False))

    client = cleaned.get("client", "Khách hàng Doanh nghiệp")
    deadline_days = int(cleaned.get("deadline_days", 10))
    orig_deadline = int(cleaned.get("orig_deadline_days", deadline_days))
    budget = float(cleaned.get("budget", 0))
    discount_pct = float(cleaned.get("discount_pct", 0))
    orig_discount = float(cleaned.get("orig_discount_pct", discount_pct))
    sla = cleaned.get("sla", "8x5 Tiêu chuẩn NBD")
    orig_sla = cleaned.get("orig_sla", sla)
    rows = cleaned.get("rows", [])

    applied_adjustments = []

    # Nếu quay lại từ Feedback Không OK: cho phép chỉnh trực tiếp hoặc áp dụng từ ghi chú / tự tối ưu
    if payload.get("adjust_discount") is not None:
        discount_pct = float(payload["adjust_discount"])
        applied_adjustments.append(f"Điều chỉnh chiết khấu thành {discount_pct}%")
    if payload.get("adjust_deadline") is not None:
        deadline_days = int(payload["adjust_deadline"])
        applied_adjustments.append(f"Điều chỉnh hạn bàn giao thành {deadline_days} ngày")
    if payload.get("adjust_sla"):
        sla = str(payload["adjust_sla"])
        applied_adjustments.append(f"Điều chỉnh SLA thành {sla}")

    if feedback_note:
        fn_lower = feedback_note.lower()
        m_disc = re.search(r"(\d+(?:\.\d+)?)\s*%", fn_lower)
        if m_disc:
            discount_pct = float(m_disc.group(1))
            applied_adjustments.append(f"Áp dụng mức chiết khấu {discount_pct}% theo ghi chú")
        elif "giảm chiết khấu" in fn_lower or "ck" in fn_lower:
            discount_pct = min(discount_pct, 5.0)
            applied_adjustments.append(f"Giảm chiết khấu về {discount_pct}% (trong quyền Sales)")
        m_dl = re.search(r"(\d+)\s*ngày", fn_lower)
        if m_dl:
            deadline_days = int(m_dl.group(1))
            applied_adjustments.append(f"Điều chỉnh hạn chót thành {deadline_days} ngày")
        if "sla chuẩn" in fn_lower or "8x5" in fn_lower:
            sla = "8x5 Tiêu chuẩn NBD"
            applied_adjustments.append("Chuyển SLA về 8x5 Tiêu chuẩn NBD")

    if auto_optimize and not applied_adjustments:
        if discount_pct > 5.0:
            discount_pct = 5.0
            applied_adjustments.append("AI tự đưa chiết khấu về 5.0% (ngưỡng Sales tự quyết)")
        if "2h" in sla.lower():
            sla = "8x5 Tiêu chuẩn NBD"
            applied_adjustments.append("AI đề xuất đổi SLA về 8x5 Tiêu chuẩn để loại bỏ phụ phí 18tr")

    feasibility_rows = []
    compliance_matrix = []
    ready_to_quote_parts = []
    must_verify_parts = []
    total_mandays = 0.0
    max_lead = 0
    hardware_cost = 0

    for r in rows:
        sku = r.get("sku", "UNRESOLVED")
        qty = int(r.get("qty", 1))
        cat = next((c for c in catalog if c["sku"] == sku), None)
        if cat:
            stock = cat["stock"]
            is_ok = stock >= qty
            lead = cat["lead_time_days"] + (0 if is_ok else 7)
            max_lead = max(max_lead, lead)
            line_total = cat["unit_price"] * qty
            hardware_cost += line_total
            manday_unit = 3.0 if "FORTI" in sku else (2.0 if "9300X" in sku else (1.0 if "9300" in sku else 0.5))
            total_mandays += manday_unit * qty

            feasibility_rows.append({
                "sku": sku,
                "name": cat["name"],
                "qty": qty,
                "stock": stock,
                "unit_price": cat["unit_price"],
                "total_price": line_total,
                "status": "Sẵn hàng" if is_ok else f"Thiếu ({stock}/{qty})",
                "is_ok": is_ok,
                "lead_days": lead
            })
            compliance_matrix.append({
                "item": cat["name"],
                "qty": qty,
                "compliance": "COMPLY (100%)" if is_ok else "PARTIAL (Nhập thêm)",
                "risk": f"ĐK bắt buộc: {r.get('mandatory', 'Đạt chuẩn')}",
                "ok": is_ok
            })
            if is_ok:
                ready_to_quote_parts.append(f"{cat['name']} (SL: {qty}) — Sẵn kho, đúng đặc tính")
            else:
                must_verify_parts.append(f"{cat['name']} thiếu tồn ({stock}/{qty}) — cần xác nhận ngày về hàng với Vận hành")
        else:
            max_lead = max(max_lead, 14)
            line_total = 20000000 * qty
            hardware_cost += line_total
            total_mandays += 2.0 * qty
            feasibility_rows.append({
                "sku": "OUT_OF_CATALOG",
                "name": r.get("clean_name", "Ngoài danh mục"),
                "qty": qty,
                "stock": 0,
                "unit_price": 20000000,
                "total_price": line_total,
                "status": "Ngoài danh mục",
                "is_ok": False,
                "lead_days": 14
            })
            compliance_matrix.append({
                "item": r.get("clean_name", "Ngoài danh mục"),
                "qty": qty,
                "compliance": "PARTIAL (Ngoài DM)",
                "risk": "Cần Kỹ thuật thẩm định nguồn OEM",
                "ok": False
            })
            must_verify_parts.append(f"Thiết bị ngoài danh mục '{r.get('clean_name')}' cần Kỹ thuật xác minh cấu hình & giá")

    # Thêm dịch vụ triển khai
    install_days = max(1, int(round(total_mandays)))
    install_cat = next((c for c in catalog if c["sku"] == "SVC-INSTALL-MANDAY"), None)
    install_unit = install_cat["unit_price"] if install_cat else 2500000
    install_total = install_unit * install_days
    feasibility_rows.append({
        "sku": "SVC-INSTALL-MANDAY",
        "name": "Dịch vụ Kỹ thuật Triển khai & Nghiệm thu (Manday)",
        "qty": install_days,
        "stock": rem_mandays,
        "unit_price": install_unit,
        "total_price": install_total,
        "status": "Đủ nhân lực" if install_days <= rem_mandays else "Quá tải kỹ sư",
        "is_ok": install_days <= rem_mandays,
        "lead_days": max_lead
    })

    sla_fee = 18000000 if "2h" in sla.lower() else 0
    service_cost = install_total + sla_fee
    total_list_price = hardware_cost + service_cost
    discount_amount = total_list_price * discount_pct / 100
    final_price = total_list_price - discount_amount

    if auto_optimize and max_lead > deadline_days:
        deadline_days = max_lead
        applied_adjustments.append(f"AI đề xuất kéo giãn hạn bàn giao thành {deadline_days} ngày để khớp lịch giao hàng")

    sched_ok = (install_days <= rem_mandays) and (max_lead <= deadline_days)
    budget_ok = (budget == 0) or (final_price <= budget)

    escalations = []
    approval_level = "Sales tự quyết"
    sales_max = approval_rules.get("sales_rep", {}).get("max_discount_pct", 5.0)
    mgr_max = approval_rules.get("sales_manager", {}).get("max_discount_pct", 10.0)

    if discount_pct > mgr_max:
        approval_level = "Ban Giám đốc phê duyệt"
        escalations.append(f"Chiết khấu {discount_pct}% vượt trần Trưởng phòng (> {mgr_max}%)")
    elif discount_pct > sales_max:
        approval_level = "Trưởng phòng Kinh doanh duyệt"
        escalations.append(f"Chiết khấu {discount_pct}% vượt quyền Sales (≤ {sales_max}%)")

    if sla_fee > 0:
        if approval_level != "Ban Giám đốc phê duyệt":
            approval_level = "Trưởng phòng Kinh doanh duyệt"
        escalations.append("SLA 24/7/2h vượt chuẩn công ty (cần duyệt chi phí đối tác)")
        must_verify_parts.append("Cam kết SLA 24/7/2h cần Vận hành xác nhận đội ứng cứu")

    if not sched_ok:
        approval_level = "Ban Giám đốc phê duyệt"
        escalations.append(f"Tiến độ ({max_lead} ngày / {install_days} công) có rủi ro so với hạn chót ({deadline_days} ngày)")

    # Kiểm tra thiết bị ngoài danh mục theo ma trận thẩm quyền
    has_out_of_catalog = any(r.get("sku") == "OUT_OF_CATALOG" for r in feasibility_rows)
    out_of_catalog_val = sum(r.get("total_price", 0) for r in feasibility_rows if r.get("sku") == "OUT_OF_CATALOG")
    if has_out_of_catalog:
        max_mgr_out = approval_rules.get("sales_manager", {}).get("max_out_of_catalog_val", 50000000)
        if out_of_catalog_val > max_mgr_out:
            approval_level = "Ban Giám đốc phê duyệt"
            escalations.append(f"Có thiết bị ngoài danh mục trị giá {out_of_catalog_val:,.0f} đ (vượt hạn mức Trưởng phòng {max_mgr_out:,.0f} đ, cần BGĐ duyệt)")
        elif approval_level != "Ban Giám đốc phê duyệt":
            approval_level = "Trưởng phòng Kinh doanh duyệt"
            escalations.append(f"Có thiết bị ngoài danh mục trị giá {out_of_catalog_val:,.0f} đ (Sales không được tự duyệt, cần Trưởng phòng duyệt nguồn)")

    if not budget_ok:
        must_verify_parts.append(f"Giá chào: {final_price:,.0f} đ | Ngân sách dự kiến của khách: {budget:,.0f} đ (cần đối chiếu lại ngân sách)")

    # Tạo Phân tích Dữ liệu Chuyên sâu & Feedback Thẩm định của AI Agent
    total_line_items = len(rows)
    total_qty = sum(int(r.get("qty", 1) or 1) for r in rows)

    # Phân loại theo nhóm thiết bị & Cơ cấu giá trị
    category_stats = {}
    for r in feasibility_rows:
        if r.get("sku") == "SVC-INSTALL-MANDAY":
            cat_name = "Dịch vụ Kỹ thuật"
        elif r.get("sku") == "OUT_OF_CATALOG":
            cat_name = "Thiết bị Ngoài danh mục (OEM)"
        else:
            cat_item = next((c for c in catalog if c["sku"] == r.get("sku")), None)
            cat_name = cat_item.get("category", "Thiết bị mạng") if cat_item else "Thiết bị mạng"

        if cat_name not in category_stats:
            category_stats[cat_name] = {"count": 0, "qty": 0, "total_value": 0}
        category_stats[cat_name]["count"] += 1
        category_stats[cat_name]["qty"] += int(r.get("qty", 1) or 1)
        category_stats[cat_name]["total_value"] += r.get("total_price", 0)

    category_breakdown = []
    tot_val = max(1, sum(v["total_value"] for v in category_stats.values()))
    for c_name, c_data in category_stats.items():
        ratio = round((c_data["total_value"] / tot_val) * 100, 1)
        category_breakdown.append({
            "category": c_name,
            "count": c_data["count"],
            "qty": c_data["qty"],
            "subtotal": c_data["total_value"],
            "ratio": ratio
        })
    category_breakdown.sort(key=lambda x: x["subtotal"], reverse=True)

    # Top 5 hạng mục chi phí lớn nhất (Pareto 80/20)
    sorted_items = sorted(
        [r for r in feasibility_rows if r.get("sku") != "SVC-INSTALL-MANDAY"],
        key=lambda x: x.get("total_price", 0),
        reverse=True
    )
    top_value_items = []
    for it in sorted_items[:5]:
        top_value_items.append({
            "name": it.get("name"),
            "qty": it.get("qty"),
            "total_price": it.get("total_price"),
            "pct": round((it.get("total_price", 0) / max(1, hardware_cost)) * 100, 1)
        })

    # Phân tích Tồn kho & Cung ứng
    hw_rows = [r for r in feasibility_rows if r.get("sku") != "SVC-INSTALL-MANDAY"]
    ready_count = len([r for r in hw_rows if r.get("is_ok")])
    shortage_rows = [r for r in hw_rows if not r.get("is_ok") and r.get("sku") != "OUT_OF_CATALOG"]
    out_of_cat_rows = [r for r in hw_rows if r.get("sku") == "OUT_OF_CATALOG"]
    stock_coverage_rate = (ready_count / max(1, len(hw_rows))) * 100.0

    # Phân tích tài chính & ngân sách
    budget_variance = (budget - final_price) if budget > 0 else 0
    budget_status = "Đạt (Tiết kiệm hơn ngân sách)" if budget_ok else f"Vượt ngân sách {abs(budget_variance):,.0f} đ"
    service_ratio = round((install_total / max(1, total_list_price)) * 100, 1)
    hardware_ratio = round((hardware_cost / max(1, total_list_price)) * 100, 1)

    # Phân tích nguồn lực kỹ thuật
    utilization_pct = min(100.0, round((install_days / max(1, rem_mandays)) * 100, 1))

    # Khuyến nghị giải pháp hành động Presales AI (Tập trung Vấn đề -> Giải pháp thực tế)
    ai_recommendations = []

    # 1. Giải pháp Tồn kho & Danh mục
    if out_of_cat_rows:
        ai_recommendations.append(
            f"🔍 **Vấn đề phát hiện:** Có {len(out_of_cat_rows)} hạng mục ngoài kho chuẩn ({out_of_catalog_val:,.0f} VNĐ). "
            f"➔ **Giải pháp đề xuất:** (1) Tư vấn khách chuyển sang SKU tương đương sẵn kho Thiên Minh (Cisco C1000/Aruba) để giao ngay trong 1-3 ngày và tối ưu 10-15% chi phí; "
            f"hoặc (2) Presales liên hệ ngay nhà phân phối OEM đặt cọc giữ giá và cam kết hạn giao cùng chứng chỉ CO/CQ chuẩn."
        )
    elif shortage_rows:
        short_names = ", ".join([r.get("name", "")[:28] for r in shortage_rows[:3]])
        ai_recommendations.append(
            f"📦 **Vấn đề phát hiện:** Thiếu tồn kho {len(shortage_rows)} hạng mục ({short_names}...). "
            f"➔ **Giải pháp đề xuất:** Đề xuất ký phụ lục giao hàng làm 2 đợt: Đợt 1 bàn giao thiết bị sẵn kho để khách nghiệm thu hạ tầng phòng server trước; "
            f"Đợt 2 bàn giao lô còn lại sau khi nhà phân phối giao hàng, tránh nguy cơ phạt hợp đồng chậm tiến độ."
        )
    else:
        ai_recommendations.append(
            f"✅ **Tồn kho sẵn sàng:** Toàn bộ thiết bị đều có sẵn tại kho Thiên Minh (Sẵn sàng giao 100%). "
            f"➔ **Giải pháp đề xuất:** Cam kết điều khoản giao hàng hỏa tốc trong 24-48 giờ để tạo lợi thế cạnh tranh quyết định trước đối thủ."
        )

    # 2. Giải pháp Tiến độ (Lead-time vs Deadline)
    if not sched_ok:
        over_days = max_lead - deadline_days
        ai_recommendations.append(
            f"⏱️ **Vấn đề phát hiện:** Thời gian cung ứng & triển khai ({max_lead} ngày) vượt hạn chót của khách ({deadline_days} ngày) {over_days} ngày. "
            f"➔ **Giải pháp đề xuất:** Đàm phán biên bản bàn giao từng phần (bàn giao thiết bị mạng Core trước trong {min(deadline_days, 7)} ngày để cấu hình hệ thống), "
            f"hoặc cấp thiết bị Demo/Loaner cho khách mượn vận hành tạm thời trong lúc chờ lô hàng chính thức."
        )
    else:
        ai_recommendations.append(
            f"⏱️ **Tiến độ đảm bảo:** Thời gian thực hiện ({max_lead} ngày) nằm trọn trong hạn mức bàn giao ({deadline_days} ngày). "
            f"➔ **Giải pháp đề xuất:** Chủ động đưa điều khoản thưởng phạt tiến độ vào hợp đồng để khẳng định năng lực và uy tín cam kết của Thiên Minh."
        )

    # 3. Giải pháp Nguồn lực Kỹ thuật (Mandays)
    if install_days > rem_mandays:
        ai_recommendations.append(
            f"🛠️ **Vấn đề phát hiện:** Khối lượng kỹ thuật ({install_days} công) vượt năng lực còn lại trong tháng ({rem_mandays} công khả dụng). "
            f"➔ **Giải pháp đề xuất:** Thuê đối tác phụ (Sub-contractor) thực hiện phần kéo cáp vật lý & bấm đầu mạng (chi phí ~8-10 triệu VNĐ), "
            f"kỹ sư Thiên Minh chỉ tập trung cấu hình Core & bàn giao để bảo đảm tiến độ mà không ảnh hưởng dự án khác."
        )
    elif utilization_pct >= 75:
        ai_recommendations.append(
            f"🛠️ **Nguồn lực kỹ thuật:** Dự án chiếm {utilization_pct}% năng lực khả dụng ({install_days}/{rem_mandays} công). "
            f"➔ **Giải pháp đề xuất:** Xếp lịch trước cho đội ngũ kỹ sư triển khai ngay khi nhận tiền tạm ứng để không bị chồng chéo lịch công tác."
        )

    # 4. Giải pháp Ngân sách & Giá chào
    if budget > 0:
        if not budget_ok:
            over_budget = final_price - budget
            pct_over = round((over_budget / budget) * 100, 1)
            ai_recommendations.append(
                f"💰 **Vấn đề phát hiện:** Giá chào ({final_price:,.0f} VNĐ) vượt ngân sách của khách ({budget:,.0f} VNĐ) {over_budget:,.0f} VNĐ (+{pct_over}%). "
                f"➔ **Giải pháp đề xuất:** Chuyển gói SLA bảo trì từ 24x7 về 8x5 (tiết kiệm ngay 18.000.000 VNĐ) hoặc xin duyệt thêm 3% chiết khấu để đưa tổng giá chào về đúng hạn mức ngân sách khách hàng."
            )
        else:
            pct_saved = round((budget_variance / budget) * 100, 1)
            ai_recommendations.append(
                f"💰 **Lợi thế ngân sách:** Giá chào ({final_price:,.0f} VNĐ) tiết kiệm hơn ngân sách khách {budget_variance:,.0f} VNĐ ({pct_saved}%). "
                f"➔ **Giải pháp đề xuất:** Tư vấn khách đầu tư thêm gói Bảo hiểm thiết bị 12 tháng hoặc bộ lưu điện UPS dự phòng để khai thác tối đa nguồn ngân sách đã được cấp."
            )
    else:
        ai_recommendations.append(
            f"💰 **Khách chưa nêu ngân sách:** "
            f"➔ **Giải pháp đề xuất:** Xuất báo giá theo 2 phương án (Phương án Tiêu Chuẩn & Phương án Nâng Cao) kèm bảng phân tích hiệu năng/chi phí để khách hàng dễ cân đối lựa chọn."
        )

    # 5. Phê duyệt & Kế hoạch chốt hợp đồng
    if approval_level != "Sales tự quyết":
        ai_recommendations.append(
            f"🛡️ **Thẩm quyền phê duyệt:** Hồ sơ thuộc thẩm quyền {approval_level}. "
            f"➔ **Giải pháp đề xuất:** File Báo cáo Word (.DOCX) đã tự động lập sẵn Mục 5 'Tờ trình Phê duyệt' với đầy đủ bảng đối chiếu chi phí & rủi ro, Sales chỉ cần ký số trình Ban Giám đốc duyệt trong 15 phút."
        )
    else:
        ai_recommendations.append(
            f"🚀 **Sales toàn quyền quyết định:** Gói thầu đạt chuẩn 100% chính sách công ty (CK ≤ 5%, đúng SLA). "
            f"➔ **Giải pháp đề xuất:** Phát hành ngay hồ sơ báo giá chính thức kèm chính sách giữ giá 15 ngày để thúc đẩy khách hàng ký hợp đồng sớm."
        )

    budget_note = (
        f"Ngân sách dự kiến của khách: {budget:,.0f} VNĐ — {'Phù hợp' if budget_ok else 'Cần đối chiếu lại'}"
        if budget > 0 else "Khách chưa nêu ngân sách dự kiến"
    )
    feedback_summary = [
        f"📊 Quy mô dữ liệu: Đã phân tích chi tiết {total_line_items} dòng hạng mục (tổng cộng {total_qty} đơn vị thiết bị).",
        f"📦 Khả năng đáp ứng kho: {ready_count}/{len(hw_rows)} hạng mục sẵn kho ({stock_coverage_rate:.1f}%). Có {len(shortage_rows)} mã cần đặt thêm và {len(out_of_cat_rows)} mã ngoài kho.",
        f"💵 Đánh giá tài chính: Giá chào sau CK {discount_pct}% là {final_price:,.0f} VNĐ ({budget_note}). Cơ cấu: {hardware_ratio}% phần cứng, {service_ratio}% dịch vụ kỹ thuật.",
        f"🛠️ Nguồn lực kỹ thuật: Cần {install_days} công kỹ thuật, chiếm {utilization_pct}% năng lực khả dụng tháng này ({rem_mandays} công còn lại).",
        f"⚖️ Phân quyền phê duyệt: {approval_level}."
    ]
    for adj in applied_adjustments:
        feedback_summary.append(f"🔧 Điều chỉnh đã áp dụng: {adj}")
    if feedback_note and not applied_adjustments:
        feedback_summary.append(f"📝 Ghi nhận phản hồi bổ sung: '{feedback_note}'.")

    loop_limit_reached = retry_count >= max_retries

    # Tạo thông báo minh bạch gửi khách hàng nếu AI/Sales có điều chỉnh so với yêu cầu gốc
    customer_change_notices = []
    if discount_pct != orig_discount:
        customer_change_notices.append(
            f"Chiết khấu thương mại: Quý khách đề nghị {orig_discount}% → Thiên Minh đề xuất áp dụng {discount_pct}% theo chính sách thương mại chuẩn."
        )
    if sla != orig_sla:
        customer_change_notices.append(
            f"Tiêu chuẩn bảo trì (SLA): Quý khách yêu cầu '{orig_sla}' → Thiên Minh đề xuất '{sla}' để tối ưu chi phí triển khai (tiết kiệm 18.000.000 VNĐ phụ phí)."
        )
    if deadline_days != orig_deadline:
        customer_change_notices.append(
            f"Thời hạn bàn giao: Quý khách mong muốn {orig_deadline} ngày → Thiên Minh đề xuất {deadline_days} ngày làm việc để đảm bảo 100% chất lượng nghiệm thu."
        )

    customer_notice_draft = ""
    if customer_change_notices:
        customer_notice_draft = f"""Kính gửi Quý Khách hàng ({client}),

Công ty Cổ phần Công nghệ Thiên Minh trân trọng cảm ơn Quý công ty đã gửi yêu cầu chào giá.
Để đảm bảo phương án khả thi tối ưu về kỹ thuật, tiến độ và chính sách giá, Thiên Minh xin phép trao đổi và đề xuất điều chỉnh một số điểm so với yêu cầu ban đầu của Quý khách như sau:
{chr(10).join([f"  - {n}" for n in customer_change_notices])}

Tổng giá trị đề xuất sau điều chỉnh là {final_price:,.0f} VNĐ (Chưa gồm VAT).
Kính mong Quý khách xem xét và xác nhận để Thiên Minh tiến hành phát hành hồ sơ hợp đồng chính thức.

Trân trọng,
Phòng Kinh doanh Dự án - Công ty Thiên Minh"""

    return {
        "client": client,
        "project_name": "Gói Thiết bị Mạng & Dịch vụ Triển khai Dự án",
        "is_sufficient": True,
        "feasibility_rows": feasibility_rows,
        "compliance_matrix": compliance_matrix,
        "ready_to_quote_parts": ready_to_quote_parts,
        "must_verify_parts": must_verify_parts,
        "feedback_summary": feedback_summary,
        "applied_adjustments": applied_adjustments,
        "customer_change_notices": customer_change_notices,
        "customer_notice_draft": customer_notice_draft,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "loop_limit_reached": loop_limit_reached,
        "hardware_cost": hardware_cost,
        "service_cost": service_cost,
        "subtotal": hardware_cost + install_total,
        "sla_fee": sla_fee,
        "total_list_price": total_list_price,
        "budget": budget,
        "budget_ok": budget_ok,
        "discount_pct": discount_pct,
        "discount_percent": discount_pct,
        "discount_amount": discount_amount,
        "base_price": total_list_price,
        "final_price": final_price,
        "max_lead_days": max_lead,
        "deadline_days": deadline_days,
        "total_mandays": install_days,
        "rem_mandays": rem_mandays,
        "sched_ok": sched_ok,
        "sla_requirement": sla,
        "requires_approval": len(escalations) > 0,
        "approval_level": approval_level,
        "escalations": escalations,
        "analytics_data": {
            "total_line_items": total_line_items,
            "total_qty": total_qty,
            "ready_count": ready_count,
            "shortage_count": len(shortage_rows),
            "out_of_cat_count": len(out_of_cat_rows),
            "stock_coverage_rate": round(stock_coverage_rate, 1),
            "category_stats": category_stats,
            "category_breakdown": category_breakdown,
            "top_value_items": top_value_items,
            "budget_variance": budget_variance,
            "budget_status": budget_status,
            "service_ratio": service_ratio,
            "hardware_ratio": hardware_ratio,
            "utilization_pct": utilization_pct,
            "ai_recommendations": ai_recommendations,
            "feedback_summary": feedback_summary
        }
    }

def parse_excel_table(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Đọc file Excel/CSV, bảo toàn 100% cấu trúc cột và dữ liệu của file gửi.
    Tự động nhận diện ngữ nghĩa các cột để AI phân tích chuẩn xác.
    """
    ext = filename.split(".")[-1].lower()
    try:
        if ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Không thể đọc file bảng tính ({ext}): {str(e)}")

    if df.empty:
        return {
            "columns": [], "raw_rows": [], "mapped_rows": [],
            "column_mapping": {}, "client": "", "budget": "0", "deadline": "10", "discount": "0", "sla": "8x5 Tiêu chuẩn NBD"
        }

    # Tìm header thực sự nếu các cột ban đầu là Unnamed: ...
    unnamed_count = sum(1 for c in df.columns if str(c).startswith("Unnamed:"))
    if unnamed_count > len(df.columns) / 2 and len(df) > 0:
        best_header_idx = -1
        for idx in range(min(5, len(df))):
            row_vals = [str(v).strip() for v in df.iloc[idx].values if pd.notna(v) and str(v).strip()]
            if len(row_vals) >= 2 and any(any(k in v.lower() for k in ["tên", "stt", "mã", "sl", "số lượng", "item", "qty", "hạng mục", "thiết bị"]) for v in row_vals):
                best_header_idx = idx
                break
        if best_header_idx >= 0:
            new_cols = [str(v).strip() if pd.notna(v) and str(v).strip() else f"Cột {i+1}" for i, v in enumerate(df.iloc[best_header_idx].values)]
            df = df.iloc[best_header_idx + 1:].reset_index(drop=True)
            df.columns = new_cols

    df = df.dropna(how="all").reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    raw_columns = [str(c).strip() if not str(c).startswith("Unnamed:") else f"Cột {i+1}" for i, c in enumerate(df.columns)]
    df.columns = raw_columns

    df = df.fillna("")
    dict_records = df.to_dict(orient="records")
    raw_records = []
    for r in dict_records:
        rec = {}
        is_empty_row = True
        for c in raw_columns:
            val = r.get(c, "")
            if isinstance(val, float):
                val = int(val) if val.is_integer() else round(val, 2)
            elif isinstance(val, int):
                val = int(val)
            else:
                val = str(val).strip()
            rec[c] = val
            if val != "":
                is_empty_row = False
        if not is_empty_row:
            raw_records.append(rec)

    cols_lower = {str(c).lower().strip(): c for c in raw_columns}

    def find_col(keywords, exclude=None):
        for kw in keywords:
            for cl, orig in cols_lower.items():
                if exclude and orig in exclude:
                    continue
                if kw in cl:
                    return orig
        return None

    col_item = find_col(["tên thiết bị", "hạng mục", "tên hàng", "mô tả hàng hóa", "sản phẩm", "item name", "item", "description", "thiết bị", "tên"])
    col_sku = find_col(["mã hàng", "mã vật tư", "part number", "part no", "p/n", "mã", "model", "sku"], exclude=[col_item] if col_item else None)
    col_qty = find_col(["số lượng", "sl", "quantity", "qty", "soluong"], exclude=[col_item, col_sku] if col_item or col_sku else None)
    col_spec = find_col(["thông số", "đặc tính", "quy cách", "cấu hình", "spec", "specification", "mô tả kỹ thuật"], exclude=[col_item, col_sku, col_qty] if col_item or col_sku or col_qty else None)
    col_mand = find_col(["ghi chú", "bắt buộc", "điều kiện", "yêu cầu", "note", "mandatory", "tiêu chuẩn"], exclude=[col_item, col_sku, col_qty, col_spec] if col_item or col_sku or col_qty or col_spec else None)
    col_price = find_col(["đơn giá", "thành tiền", "giá", "price", "amount", "dự toán"], exclude=[col_item, col_sku, col_qty, col_spec, col_mand] if col_item or col_sku or col_qty or col_spec or col_mand else None)
    col_unit = find_col(["đvt", "đơn vị tính", "đơn vị", "unit"], exclude=[col_item, col_sku, col_qty, col_spec, col_mand, col_price] if col_item or col_sku or col_qty or col_spec or col_mand or col_price else None)

    if not col_item and raw_columns:
        for c in raw_columns:
            if str(c).lower() in ["stt", "no", "id", "#"]:
                continue
            col_item = c
            break
        if not col_item:
            col_item = raw_columns[0]

    mapped_rows: List[SheetRow] = []
    total_est_cost = 0

    for rec in raw_records:
        item_val = str(rec.get(col_item, "")).strip() if col_item else ""
        sku_val = str(rec.get(col_sku, "")).strip() if col_sku else ""
        spec_val = str(rec.get(col_spec, "")).strip() if col_spec else ""
        if sku_val and sku_val.lower() not in item_val.lower():
            combined_item = f"{item_val} ({sku_val})" if item_val else sku_val
        else:
            combined_item = item_val

        qty_val = str(rec.get(col_qty, "1")).strip() if col_qty else "1"
        q_digits = re.findall(r"\d+", qty_val)
        clean_qty = q_digits[0] if q_digits else "1"

        mand_val = str(rec.get(col_mand, "")).strip() if col_mand else ""
        unit_val = str(rec.get(col_unit, "")).strip() if col_unit else ""
        if unit_val:
            spec_val = f"ĐVT: {unit_val} | {spec_val}" if spec_val else f"ĐVT: {unit_val}"

        if col_price:
            pr_raw = str(rec.get(col_price, "0")).replace(",", "").replace(".", "").replace("đ", "").strip()
            pr_digits = re.findall(r"\d+", pr_raw)
            if pr_digits:
                try:
                    line_pr = float(pr_digits[0])
                    total_est_cost += line_pr * (int(clean_qty) if "đơn giá" in str(col_price).lower() else 1)
                except Exception:
                    pass

        if combined_item:
            mapped_rows.append(SheetRow(
                item=combined_item,
                spec=spec_val or "Theo thông số file Excel",
                qty=clean_qty,
                mandatory=mand_val or "Theo tiêu chuẩn chào thầu",
                raw_data=rec
            ))

    base_name = os.path.splitext(os.path.basename(filename))[0]
    clean_client_from_file = base_name.replace("_", " ").replace("-", " ").strip().title()
    budget_str = str(int(total_est_cost)) if total_est_cost > 0 else "0"

    return {
        "columns": raw_columns,
        "raw_rows": raw_records,
        "column_mapping": {
            "item": col_item or "",
            "sku": col_sku or "",
            "spec": col_spec or "",
            "qty": col_qty or "",
            "mandatory": col_mand or "",
            "price": col_price or "",
            "unit": col_unit or ""
        },
        "mapped_rows": mapped_rows,
        "client": clean_client_from_file,
        "budget": budget_str,
        "deadline": "10",
        "discount": "0",
        "sla": "8x5 Tiêu chuẩn NBD"
    }

@app.post("/api/upload-sheet")
async def api_upload_sheet(file: UploadFile = File(...)):
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()
    if ext in ["xlsx", "xls", "csv"]:
        try:
            parsed = parse_excel_table(file_bytes, file.filename)
            return {
                "ok": True,
                "is_excel": True,
                "filename": file.filename,
                "columns": parsed["columns"],
                "raw_rows": parsed["raw_rows"],
                "column_mapping": parsed["column_mapping"],
                "rows": [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in parsed["mapped_rows"]],
                "client": parsed["client"],
                "budget": parsed["budget"],
                "deadline": parsed["deadline"],
                "discount": parsed["discount"],
                "sla": parsed["sla"]
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        txt = extract_text_from_file(file_bytes, file.filename)
        return {
            "ok": True,
            "is_excel": False,
            "filename": file.filename,
            "columns": ["Hạng mục", "Thông số", "Số lượng", "Điều kiện"],
            "raw_rows": [{"Hạng mục": txt[:120], "Thông số": "Trích xuất từ tài liệu", "Số lượng": "1", "Điều kiện": ""}],
            "rows": [{"item": txt[:120], "spec": "Trích xuất từ tài liệu", "qty": "1", "mandatory": ""}]
        }

@app.post("/api/ingest-raw-file")
async def api_ingest_raw_file(file: UploadFile = File(...)):
    """
    Nhận bất kỳ file dữ liệu raw nào (.pdf, .docx, .xlsx, .csv, .txt),
    bóc tách theo đúng định dạng file gốc, kiểm tra định dạng và làm sạch.
    """
    config = load_config()
    catalog = config.get("catalog", [])
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()

    raw_preview = ""
    client = ""
    deadline = ""
    budget = "0"
    discount = "0"
    sla = "8x5 Tiêu chuẩn NBD"
    rows: List[SheetRow] = []
    columns: List[str] = []
    raw_records: List[Dict[str, Any]] = []
    column_mapping: Dict[str, str] = {}
    is_excel = ext in ["xlsx", "xls", "csv"]

    if is_excel:
        try:
            parsed = parse_excel_table(file_bytes, file.filename)
            columns = parsed["columns"]
            raw_records = parsed["raw_rows"]
            column_mapping = parsed["column_mapping"]
            client = parsed["client"]
            deadline = parsed["deadline"]
            budget = parsed["budget"]
            discount = parsed["discount"]
            sla = parsed["sla"]
            rows = parsed["mapped_rows"]
            raw_preview = f"File bảng tính: {file.filename}\nCột theo file: {', '.join(columns)}\nTổng số dòng: {len(raw_records)}"
        except Exception as e:
            return {"ok": False, "error": f"Không đọc được file bảng tính: {str(e)}"}
    else:
        raw_preview = extract_text_from_file(file_bytes, file.filename)
        extracted = extract_rfp_with_gemini(raw_preview, catalog)
        client = extracted.get("client_name") or os.path.splitext(file.filename)[0]
        deadline = str(extracted.get("deadline_days", 0) or "")
        budget = str(int(extracted.get("budget_vnd", 0))) if extracted.get("budget_vnd") else "0"
        discount = str(extracted.get("discount_requested_pct", 0))
        sla = extracted.get("sla_requirement", "8x5 Tiêu chuẩn NBD")
        for it in extracted.get("items", []):
            if it.get("sku") == "SVC-INSTALL-MANDAY":
                continue
            rows.append(SheetRow(
                item=str(it.get("name", "")),
                spec=str(it.get("spec") or "Trích xuất từ tài liệu raw"),
                qty=str(it.get("qty", 1)),
                mandatory=str(it.get("mandatory") or "")
            ))
        columns = ["Hạng mục", "Thông số", "Số lượng", "Điều kiện"]
        raw_records = [{"Hạng mục": r.item, "Thông số": r.spec, "Số lượng": r.qty, "Điều kiện": r.mandatory} for r in rows]

    sheet_payload = SheetPayload(
        client=client,
        deadline=deadline,
        budget=budget,
        discount=discount,
        sla=sla,
        rows=rows,
        columns=columns,
        raw_rows=raw_records
    )
    validation = await api_validate_sheet(sheet_payload)
    cleaning = await api_clean_sheet(sheet_payload) if validation["valid_format"] else None

    return {
        "ok": True,
        "filename": file.filename,
        "is_excel": is_excel,
        "columns": columns,
        "raw_rows": raw_records,
        "column_mapping": column_mapping,
        "raw_preview": raw_preview[:2500],
        "extracted_sheet": sheet_payload.model_dump(),
        "validation": validation,
        "cleaning": cleaning
    }

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/khach-hang", response_class=HTMLResponse)
async def serve_customer_portal():
    cust_path = os.path.join(STATIC_DIR, "customer.html")
    with open(cust_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

class CustomerSubmitPayload(SheetPayload):
    contact_email: str = ""

class GoogleSheetSyncPayload(BaseModel):
    sheet_url: str

@app.post("/api/customer/submit")
async def api_customer_submit(payload: CustomerSubmitPayload):
    val = await api_validate_sheet(payload)
    if not val["valid_format"]:
        return {
            "accepted": False,
            "title": "❌ Dữ liệu nhập sai định dạng — Vui lòng sửa lại:",
            "errors": val["errors"]
        }
    clean = await api_clean_sheet(payload)
    if not clean["is_clean"]:
        return {
            "accepted": False,
            "title": "⚠️ Dữ liệu chưa đạt chuẩn để phân tích — Vui lòng làm rõ:",
            "errors": clean["unclean_issues"]
        }
    code = save_customer_request(
        client_name=clean["cleaned_data"]["client"],
        contact_email=payload.contact_email.strip(),
        source="PORTAL_KH",
        raw_sheet=payload.model_dump(),
        cleaned_data=clean["cleaned_data"]
    )
    return {"accepted": True, "request_code": code}

@app.get("/api/inbound-requests")
async def api_inbound_requests():
    return {"requests": get_customer_requests()}

@app.post("/api/inbound-requests/{code}/processed")
async def api_mark_inbound_processed(code: str):
    mark_customer_request_status(code, "PROCESSED")
    return {"ok": True, "requests": get_customer_requests()}

@app.post("/api/webhook/google-sheet")
async def api_webhook_google_sheet(payload: CustomerSubmitPayload):
    return await api_customer_submit(payload)

@app.post("/api/sync-google-sheet")
async def api_sync_google_sheet(req: GoogleSheetSyncPayload):
    url = req.sheet_url.strip()
    if "docs.google.com/spreadsheets/d/" in url and "/export" not in url:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
        if m:
            url = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    try:
        csv_bytes = urllib.request.urlopen(url, timeout=10).read()
        df = pd.read_csv(io.BytesIO(csv_bytes))
        if df.empty:
            return {"ok": False, "error": "Google Sheet không có dữ liệu."}
        # Hỗ trợ sheet có cột: client, deadline, budget, discount, sla, item, spec, qty, mandatory
        first = df.iloc[0].to_dict()
        client = str(first.get("client") or first.get("Khách hàng") or "Khách hàng từ Google Sheet")
        deadline = str(first.get("deadline") or first.get("Hạn") or "10")
        budget = str(first.get("budget") or first.get("Ngân sách") or "0")
        discount = str(first.get("discount") or first.get("Chiết khấu") or "0")
        sla = str(first.get("sla") or first.get("SLA") or "8x5 Tiêu chuẩn NBD")
        rows = []
        df_records = df.fillna("").to_dict(orient="records")
        for r in df_records:
            vals = list(r.values())
            rows.append(SheetRow(
                item=str(r.get("item") or r.get("Hạng mục") or (vals[0] if len(vals) > 0 else "") or ""),
                spec=str(r.get("spec") or r.get("Thông số") or (vals[1] if len(vals) > 1 else "") or ""),
                qty=str(r.get("qty") or r.get("Số lượng") or (vals[2] if len(vals) > 2 else "1") or "1"),
                mandatory=str(r.get("mandatory") or r.get("Điều kiện bắt buộc") or (vals[3] if len(vals) > 3 else "") or "")
            ))
        payload = SheetPayload(client=client, deadline=deadline, budget=budget, discount=discount, sla=sla, rows=rows)
        val = await api_validate_sheet(payload)
        if not val["valid_format"]:
            return {"ok": False, "error": "Lỗi định dạng trên Sheet: " + "; ".join(val["errors"])}
        clean = await api_clean_sheet(payload)
        if not clean["is_clean"]:
            return {"ok": False, "error": "Dữ liệu trên Sheet chưa sạch: " + "; ".join(clean["unclean_issues"])}
        code = save_customer_request(
            client_name=clean["cleaned_data"]["client"],
            contact_email="",
            source="GOOGLE_SHEET",
            raw_sheet=payload.model_dump(),
            cleaned_data=clean["cleaned_data"]
        )
        return {"ok": True, "request_code": code, "requests": get_customer_requests()}
    except Exception as e:
        return {"ok": False, "error": f"Không đọc được Google Sheet (hãy bật quyền 'Anyone with the link - Viewer'): {str(e)}"}

@app.get("/api/bootstrap")
async def api_bootstrap():
    config = load_config()
    ops = config.get("operations_capacity", {"total_capacity_mandays": 110, "booked_mandays_current_month": 68})
    return {
        "ai_connected": bool(GEMINI_API_KEY),
        "catalog": config.get("catalog", []),
        "approval_rules": config.get("approval_rules", {}),
        "operations_capacity": ops,
        "rem_mandays": ops.get("total_capacity_mandays", 110) - ops.get("booked_mandays_current_month", 68),
        "scenarios": SHEET_SCENARIOS,
        "dossiers": get_all_dossiers(),
        "inbound_requests": get_customer_requests()
    }

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    config = load_config()
    reply = chat_with_assistant(req.message, req.history, req.context, config)
    return {"reply": reply}

@app.post("/api/save-dossier")
async def api_save_dossier(req: SaveDossierRequest):
    ev = req.evaluation
    code = save_dossier({
        "client_name": ev.get("client", "Khách hàng"),
        "project_name": ev.get("project_name", "Dự án Mạng"),
        "deadline_days": ev.get("deadline_days", 0),
        "discount_requested": ev.get("discount_pct", 0),
        "total_list_price": ev.get("total_list_price", 0),
        "final_bid_price": ev.get("final_price", 0),
        "approval_level": ev.get("approval_level", ""),
        "status": req.status,
        "compliance_summary": ev.get("compliance_matrix", []),
        "pricing_details": ev,
        "approval_notes": req.notes
    })
    return {"dossier_code": code, "dossiers": get_all_dossiers()}

@app.post("/api/export-docx")
async def api_export_docx(evaluation: Dict[str, Any]):
    comp_matrix = [
        {
            "Hạng mục": c.get("item", c.get("Hạng mục", "")),
            "Số lượng": c.get("qty", c.get("Số lượng", 1)),
            "Đáp ứng (Compliance)": c.get("compliance", c.get("Đáp ứng (Compliance)", "")),
            "Đánh giá Rủi ro": c.get("risk", c.get("Đánh giá Rủi ro", ""))
        }
        for c in evaluation.get("compliance_matrix", [])
    ]
    raw_rows = evaluation.get("pricing_rows") or evaluation.get("feasibility_rows", [])
    pricing_rows = [
        {
            "Tên Hạng mục": r.get("name", r.get("Tên Hạng mục", "")),
            "SL Yêu cầu": r.get("qty", r.get("SL Yêu cầu", 1)),
            "Đơn giá niêm yết (VNĐ)": r.get("unit_price", r.get("Đơn giá niêm yết (VNĐ)", 0)),
            "Thành tiền (VNĐ)": r.get("total_price", r.get("Thành tiền (VNĐ)", 0))
        }
        for r in raw_rows
    ]
    bio = generate_bid_docx({
        "title": evaluation.get("title", "HỒ SƠ ĐỀ XUẤT GIẢI PHÁP & BẢN CHÀO GIÁ DỰ ÁN"),
        "client": evaluation.get("client", "Quý Khách hàng"),
        "project_name": evaluation.get("project_name", "Gói Thiết bị Mạng & Dịch vụ"),
        "date_str": evaluation.get("date_str", ""),
        "compliance_matrix": comp_matrix,
        "pricing_rows": pricing_rows,
        "total_list_price": evaluation.get("total_list_price", 0),
        "discount_pct": evaluation.get("discount_pct", 0),
        "discount_amount": evaluation.get("discount_amount", 0),
        "final_price": evaluation.get("final_price", 0),
        "deadline_days": evaluation.get("deadline_days", 10),
        "sla_requirement": evaluation.get("sla_requirement", "8x5 Tiêu chuẩn NBD"),
        "customer_change_notices": evaluation.get("customer_change_notices", []),
        "notes": evaluation.get("notes", ""),
        "signer_client": evaluation.get("signer_client", ""),
        "signer_company": evaluation.get("signer_company", "")
    })
    return StreamingResponse(
        bio,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=Bao_Gia_ThienMinh.docx"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
