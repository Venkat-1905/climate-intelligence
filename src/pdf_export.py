"""
Climate Intelligence System — Executive PDF Export
===================================================
Scrapes current global KPIs and top risk narratives,
and builds an auto-generated 1-page business report 
using fpdf2.

Outputs:
  - outputs/Climate_Executive_Summary.pdf
"""

import os
import sys
import pandas as pd
from fpdf import FPDF
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


class PDFReport(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(6, 214, 160) # Green accent
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Climate Intelligence Executive Summary', border=0, align='C', new_x="RIGHT", new_y="TOP")
        self.ln(15)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        # Page number
        self.cell(0, 10, f'Generated automatically by Climate AI Engine ({datetime.now().strftime("%Y-%m-%d")}) - Page {self.page_no()}', border=0, align='C', new_x="RIGHT", new_y="TOP")

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f'  {title}', border=0, align='L', fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        # remove unicode dashes for helvetica core fonts
        text = text.replace("—", "-").replace("₂", "2")
        self.multi_cell(0, 6, text)
        self.ln()

def generate_pdf():
    print("=" * 60)
    print("📄 CLIMATE INTELLIGENCE - PDF GENERATOR")
    print("=" * 60)
    
    # Load required data
    fact_path = os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv")
    narrative_path = os.path.join(OUTPUT_DIR, "narratives.csv")
    
    if not os.path.exists(fact_path) or not os.path.exists(narrative_path):
        print("❌ Required datasets not found. Run pipeline and narratives first.")
        return False
        
    fact = pd.read_csv(fact_path)
    narratives = pd.read_csv(narrative_path)
    
    latest_year = fact["year"].max()
    latest = fact[fact["year"] == latest_year]
    total_co2 = latest["co2"].sum() / 1000
    avg_temp = latest["temperature_anomaly"].mean()
    
    # Generate content
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.chapter_title(f"Global Impact Overview ({latest_year})")
    body1 = (
        f"Global CO2 emissions reached an estimated {total_co2:.1f} Gigatonnes across tracked nations. "
        f"The global temperature anomaly averaged {avg_temp:.2f}C above pre-industrial baselines. "
        f"Immediate action across energy sectors remains critical as the transition to renewables accelerates."
    )
    pdf.chapter_body(body1)
    
    pdf.chapter_title("Critical Risk Profiles (Top 5 Vulnerable Nations)")
    
    # Top 5 Narratives
    narratives = narratives.sort_values("risk_score", ascending=False).head(5)
    for _, row in narratives.iterrows():
        country = row["country"]
        narrative_text = str(row["narrative"]).encode('utf-8', 'ignore').decode('utf-8')
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_text_color(239, 68, 68) # Red alert for high risk
        pdf.cell(0, 6, f"> {country}", border=0, align='L', new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        pdf.chapter_body(narrative_text)
        
    # Recommendations snippet
    pdf.chapter_title("Macro Policy Recommendations")
    pdf.chapter_body(
        "- Rapidly phase out coal dependency in major emitting nations.\n"
        "- Increase adoption of renewable energy by minimum 15% to stabilize risk trajectories.\n"
        "- Provide direct funding and physical adaptation resources to states with >0.5 ND-GAIN vulnerability indices."
    )

    out_file = os.path.join(OUTPUT_DIR, "Climate_Executive_Summary.pdf")
    pdf.output(out_file)
    print(f"   ✅ Auto-generated 1-page PDF summary: {out_file}")
    
    return True


if __name__ == "__main__":
    generate_pdf()
