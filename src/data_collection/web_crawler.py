"""
Module: web_crawler.py
Bài 2 (Lecture 2): Thu thập dữ liệu Web bằng Crawler (Web Scraping).

Cào bảng "List of the busiest airports in the United States" trên Wikipedia bằng requests + BeautifulSoup
để lấy mức độ đông đúc (số lượt hành khách lên máy bay / năm, nguồn FAA) của các sân bay hub Mỹ.
Dữ liệu này được tích hợp vào tập chuyến bay theo khóa sân bay đến (dest = IATA code), tạo đặc trưng:
  - dest_hub_tier: 2 = Large hub, 1 = Medium hub, 0 = không thuộc 2 nhóm trên
  - dest_enplanements_m: số lượt khách lên máy bay (triệu) của sân bay đến

Lưu ý: bảng Wikipedia chỉ có số liệu từ năm 2016 trở đi, nên năm 2016 (gần 2013 nhất) được dùng làm
đại diện cho quy mô sân bay đến. Đây là giới hạn cần nêu trong báo cáo.

Trang HTML gốc được lưu lại (data/raw/wiki_busiest_us_airports.html) để có thể chạy lại khi không có mạng.
"""

import os
import re
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_the_United_States"
HTTP_HEADERS = {
    'User-Agent': 'NMKHDL-coursework-crawler/1.0 (educational project)',
    'Accept-Language': 'en;q=0.9'
}
REFERENCE_YEAR = 2016  # Năm cũ nhất có trong bảng, gần với dữ liệu chuyến bay 2013 nhất


class AirportTrafficCrawler:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.html_path = os.path.join(output_dir, "wiki_busiest_us_airports.html")
        self.csv_path = os.path.join(output_dir, "crawled_us_airport_traffic.csv")
        os.makedirs(output_dir, exist_ok=True)

    def fetch_html(self, use_cache: bool = False) -> str:
        """Tải trang Wikipedia (hoặc đọc bản HTML đã lưu nếu use_cache=True)."""
        if use_cache and os.path.exists(self.html_path):
            print(f"[Web Crawler] Đọc trang đã lưu: {self.html_path}")
            with open(self.html_path, encoding='utf-8') as f:
                return f.read()
        print(f"[Web Crawler] Đang tải: {WIKI_URL}")
        resp = requests.get(WIKI_URL, headers=HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
        with open(self.html_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        print(f"[Web Crawler] Đã lưu HTML gốc ({len(resp.text):,} ký tự) vào: {self.html_path}")
        return resp.text

    @staticmethod
    def _to_int(text: str):
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else None

    @staticmethod
    def _parse_hub_table(table, tier: int, tier_name: str) -> pd.DataFrame:
        """Đọc một bảng wikitable (Large hubs / Medium hubs) thành DataFrame."""
        rows = table.find_all('tr')
        header = [c.get_text(' ', strip=True) for c in rows[0].find_all(['th', 'td'])]
        # Cột năm có dạng "2025 [ 2 ]" -> lấy 4 chữ số đầu
        year_cols = {i: int(m.group(1)) for i, h in enumerate(header) if (m := re.match(r"(\d{4})", h))}
        iata_idx = next(i for i, h in enumerate(header) if 'IATA' in h)
        name_idx = next(i for i, h in enumerate(header) if h.lower().startswith('airport'))
        state_idx = next((i for i, h in enumerate(header) if h.lower() == 'state'), None)

        records = []
        for tr in rows[1:]:
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['th', 'td'])]
            if len(cells) < len(header):
                continue
            iata = cells[iata_idx].strip().upper()
            if not re.fullmatch(r"[A-Z]{3}", iata):
                continue
            rec = {
                'iata': iata,
                'airport_name': cells[name_idx],
                'state': cells[state_idx] if state_idx is not None else None,
                'hub_tier': tier,
                'hub_category': tier_name,
            }
            for i, year in year_cols.items():
                rec[f'enplanements_{year}'] = AirportTrafficCrawler._to_int(cells[i])
            records.append(rec)
        return pd.DataFrame(records)

    def crawl_airport_traffic(self, use_cache: bool = False) -> pd.DataFrame:
        html = self.fetch_html(use_cache=use_cache)
        soup = BeautifulSoup(html, 'html.parser')

        frames = []
        for table in soup.select('table.wikitable'):
            section = table.find_previous(['h2', 'h3'])
            title = section.get_text(strip=True).lower() if section else ''
            if title.startswith('large hub'):
                frames.append(self._parse_hub_table(table, 2, 'Large hub'))
            elif title.startswith('medium hub'):
                frames.append(self._parse_hub_table(table, 1, 'Medium hub'))

        if not frames:
            raise ValueError("Không tìm thấy bảng 'Large hubs' / 'Medium hubs' trên trang - cấu trúc trang có thể đã đổi.")

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset='iata')
        df.to_csv(self.csv_path, index=False)
        print(f"[Web Crawler - Hoàn tất] Cào được {len(df)} sân bay hub "
              f"({(df.hub_tier == 2).sum()} large, {(df.hub_tier == 1).sum()} medium) -> {self.csv_path}")
        return df


if __name__ == "__main__":
    crawler = AirportTrafficCrawler()
    crawler.crawl_airport_traffic(use_cache="--offline" in sys.argv)
