import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets APIの設定
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# サービスアカウントキーの JSON ファイルのパスを指定
#credentials = ServiceAccountCredentials.from_json_keyfile_name("path/to/your-service-account-key.json", scope)
credentials_info = st.secrets["gcp_service_account"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
client = gspread.authorize(credentials)

# スプレッドシートを開く
spreadsheet_id = st.secrets["SPREADSHEET_ID"]  # スプレッドシートIDを設定
sheet_name = "tanabata_wishes"  # シート名を設定
sheet = client.open_by_key(spreadsheet_id).worksheet(sheet_name)


def write_wish(data):
    """願い事をGoogleスプレッドシートに書き込む"""
    sheet.append_row([
        data["age"],
        data["gender"],
        data["country"],
        data["prefecture"],
        data["wish_original"],
        data["wish_ja"],
        data["wish_en"],
        data["wish_zh"],
        data["wish_ko"],
        data["flag"],
        data.get("timestamp", "")
    ])


def read_wishes(num_wishes):
    """Googleスプレッドシートから指定した数の願い事をランダムに読み込む"""
    records = sheet.get_all_records()
    import random
    if len(records) > num_wishes:
        records = random.sample(records, num_wishes)
    return records
