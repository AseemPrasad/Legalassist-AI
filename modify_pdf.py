import os
import structlog
from datetime import datetime

file_path = "pdf_exporter.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace all occurrences of 'Arial' with 'Times'
content = content.replace("'Arial'", "'Times'")

# Replace Header, Footer, Chapter Title
old_branding = """    def header(self):
        """Add header to each page"""
        # Gradient background for header
        self.set_fill_color(45, 45, 255)
        self.rect(0, 0, 210, 20, 'F')

        # Title
        self.set_font('Times', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 7)
        self.cell(0, 6, 'LegalAssist AI - Case Summary', align='C')

        # Page number
        self.set_font('Times', 'I', 8)
        self.set_xy(190, 15)
        self.cell(0, 4, f'Page {self.page_no()}', align='R')

        # Reset text color
        self.set_text_color(0, 0, 0)
        self.set_xy(10, 25)

    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%d %B %Y")} | LegalAssist AI', align='C')

    def chapter_title(self, label):
        """Add chapter title"""
        self.set_fill_color(26, 26, 46)
        self.set_text_color(255, 255, 255)
        self.set_font('Times', 'B', 12)
        self.cell(0, 8, label, 0, 1, 'L', True)
        self.ln(4)
        self.set_text_color(0, 0, 0)"""

new_branding = """    def header(self):
        """Add header to each page"""
        self.set_font('Times', 'B', 14)
        self.cell(0, 8, 'LEGALASSIST AI - CASE BRIEFING', 0, 1, 'C')
        self.set_font('Times', 'I', 10)
        self.cell(0, 5, 'STRICTLY CONFIDENTIAL', 0, 1, 'C')
        self.line(10, 25, 200, 25)
        self.line(10, 26, 200, 26)
        self.ln(10)

    def footer(self):
        """Add footer to each page"""
        self.set_y(-20)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_font('Times', 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Generated on {datetime.now().strftime("%d %B %Y")}', align='L')
        self.set_xy(160, self.get_y() - 5)
        self.cell(0, 5, f'Page {self.page_no()}', align='R')

    def chapter_title(self, label):
        """Add chapter title"""
        self.ln(5)
        self.set_font('Times', 'B', 12)
        self.cell(0, 6, label.upper(), 0, 1, 'L')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)"""

# Validate and replace with explicit feedback
if old_branding in content:
    content = content.replace(old_branding, new_branding)
    logger.info("branding_replaced", msg="Branding template replaced successfully.")
else:
    logger.warning("branding_target_missing", msg="old_branding not found - already refactored or moved.")
    raise ValueError("Branding replacement target not found in pdf_exporter.py. Migration aborted.")

# Replace Status Badge
old_status = """        # Status badge
        status = case['status'].upper()
        status_color = {
            'ACTIVE': (0, 200, 83),
            'APPEALED': (255, 145, 0),
            'CLOSED': (97, 97, 97),
            'PENDING': (41, 182, 246),
        }.get(status, (128, 128, 128))

        pdf.set_fill_color(*status_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Times', 'B', 10)
        pdf.cell(30, 6, status, 0, 1, 'C', True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)"""

new_status = """        # Status
        status = case['status'].upper()
        pdf.set_font('Times', 'B', 10)
        pdf.cell(0, 6, f"STATUS: {status}", 0, 1, 'C')
        pdf.line(10, pdf.get_y() + 5, 200, pdf.get_y() + 5)
        pdf.ln(10)"""

if old_status in content:
    content = content.replace(old_status, new_status)
    logger.info("status_badge_replaced", msg="Status badge replaced successfully.")
else:
    logger.warning("status_target_missing", msg="old_status not found - already refactored.")
    raise ValueError("Status badge replacement target not found in pdf_exporter.py. Migration aborted.")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

logger = structlog.get_logger(__name__)
logger.info("pdf_exporter_modified", msg="PDF Exporter modified for legal format.")
