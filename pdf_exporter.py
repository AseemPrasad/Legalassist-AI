"""
PDF Export functionality for LegalAssist AI.
Generate professional PDF case summaries for export and sharing.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import os
import logging

from fpdf import FPDF

from database import SessionLocal, Case, CaseDocument, CaseDeadline, CaseTimeline, CaseStatus
from case_manager import get_case_detail, generate_case_summary_text

logger = logging.getLogger(__name__)


class LegalAssistPDF(FPDF):
    """Custom PDF class with LegalAssist branding"""

    def _clean(self, txt):
        if not isinstance(txt, str):
            return txt
        replacements = {
            '•': '-', '–': '-', '—': '-', 
            '\u201c': '"', '\u201d': '"', 
            '\u2018': "'", '\u2019': "'", 
            '…': '...'
        }
        for k, v in replacements.items():
            txt = txt.replace(k, v)
        return txt.encode('latin-1', 'replace').decode('latin-1')

    def cell(self, w, h=0, txt="", *args, **kwargs):
        txt = self._clean(txt)
        super().cell(w, h, txt, *args, **kwargs)

    def multi_cell(self, w, h, txt, *args, **kwargs):
        txt = self._clean(txt)
        super().multi_cell(w, h, txt, *args, **kwargs)

    def header(self):
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
        self.ln(4)

    def chapter_body(self, text):
        """Add chapter body text"""
        self.set_font('Times', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(5)

    def add_table_row(self, labels, values, widths=None):
        """Add a table row with labels and values"""
        if widths is None:
            widths = [50, 140]

        x_start = self.get_x()
        y_start = self.get_y()

        # Background
        self.set_fill_color(240, 240, 240)
        self.rect(x_start, y_start, sum(widths), 7, 'F')

        # Labels
        self.set_font('Times', 'B', 9)
        for i, label in enumerate(labels):
            self.cell(widths[i], 7, label, 1, 0, 'L', False)

        self.ln(7)

        # Values
        for i, value in enumerate(values):
            self.set_font('Times', '', 9)
            self.cell(widths[i], 7, str(value)[:50], 1, 0, 'L', False)

        self.ln(7)


def generate_case_pdf(user_id: int, case_id: int) -> Optional[bytes]:
    """
    Generate a PDF summary of a case.
    Returns PDF as bytes or None if failed.
    """
    db = SessionLocal()
    try:
        # Get case data
        case_data = get_case_detail(user_id, case_id)

        if not case_data:
            logger.error(f"Case {case_id} not found or access denied for user {user_id}")
            return None

        case = case_data["case"]
        documents = case_data["documents"]
        timeline = case_data["timeline"]
        deadlines = case_data["deadlines"]
        remedies = case_data.get("remedies")

        # Create PDF
        pdf = LegalAssistPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ==================== CASE HEADER ====================
        pdf.set_font('Times', 'B', 18)
        pdf.cell(0, 10, case.get('title') or case['case_number'], 0, 1, 'C')

        pdf.set_font('Times', '', 11)
        pdf.cell(0, 8, f"Case No: {case['case_number']}", 0, 1, 'C')
        pdf.ln(5)

        # Status
        status = case['status'].upper()
        pdf.set_font('Times', 'B', 10)
        pdf.cell(0, 6, f"STATUS: {status}", 0, 1, 'C')
        pdf.line(10, pdf.get_y() + 5, 200, pdf.get_y() + 5)
        pdf.ln(10)

        # ==================== CASE INFORMATION ====================
        pdf.chapter_title('Case Information')

        info_items = [
            ('Case Type:', case['case_type'].title()),
            ('Jurisdiction:', case['jurisdiction']),
            ('Status:', case['status'].title()),
            ('Created:', datetime.fromisoformat(case['created_at']).strftime('%d %B %Y')),
        ]

        for label, value in info_items:
            pdf.set_font('Times', 'B', 10)
            pdf.cell(40, 6, label, 0, 0)
            pdf.set_font('Times', '', 10)
            pdf.cell(0, 6, value, 0, 1)

        pdf.ln(5)

        # ==================== REMEDIES & ADVICE ====================
        if remedies:
            pdf.add_page()
            pdf.chapter_title('Legal Remedies & Advice')

            pdf.set_font('Times', 'B', 10)
            pdf.cell(0, 6, 'What Happened:', 0, 1)
            pdf.set_font('Times', '', 10)
            pdf.multi_cell(0, 5, remedies.get('what_happened', 'N/A'))
            pdf.ln(3)

            pdf.set_font('Times', 'B', 10)
            pdf.cell(0, 6, 'Can You Appeal:', 0, 1)
            pdf.set_font('Times', '', 10)
            pdf.multi_cell(0, 5, remedies.get('can_appeal', 'N/A'))
            pdf.ln(3)

            # Appeal details in columns
            col_width = 60

            if remedies.get('appeal_days'):
                pdf.set_font('Times', 'B', 10)
                pdf.cell(col_width, 6, 'Appeal Timeline:', 0, 0)
                pdf.set_font('Times', '', 10)
                pdf.cell(0, 6, remedies.get('appeal_days', 'N/A'), 0, 1)

            if remedies.get('appeal_court'):
                pdf.set_font('Times', 'B', 10)
                pdf.cell(col_width, 6, 'Appeal Court:', 0, 0)
                pdf.set_font('Times', '', 10)
                pdf.cell(0, 6, remedies.get('appeal_court', 'N/A'), 0, 1)

            if remedies.get('cost_estimate'):
                pdf.set_font('Times', 'B', 10)
                pdf.cell(col_width, 6, 'Estimated Cost:', 0, 0)
                pdf.set_font('Times', '', 10)
                pdf.cell(0, 6, remedies.get('cost_estimate', 'N/A'), 0, 1)

            if remedies.get('first_action'):
                pdf.ln(3)
                pdf.set_font('Times', 'B', 10)
                pdf.cell(0, 6, 'First Action:', 0, 1)
                pdf.set_font('Times', '', 10)
                pdf.multi_cell(0, 5, remedies.get('first_action', 'N/A'))

            if remedies.get('deadline'):
                pdf.ln(3)
                pdf.set_font('Times', 'B', 10)
                pdf.cell(0, 6, 'Important Deadline:', 0, 1)
                pdf.set_font('Times', '', 10)
                pdf.set_text_color(255, 0, 0)
                pdf.multi_cell(0, 5, remedies.get('deadline', 'N/A'))
                pdf.set_text_color(0, 0, 0)

        # ==================== DOCUMENTS ====================
        pdf.add_page()
        pdf.chapter_title('Documents')

        if documents:
            for i, doc in enumerate(documents):
                pdf.set_font('Times', 'B', 11)
                pdf.cell(0, 6, f"{i+1}. {doc['document_type']}", 0, 1)

                pdf.set_font('Times', 'I', 9)
                upload_date = datetime.fromisoformat(doc['uploaded_at']).strftime('%d %b %Y')
                pdf.cell(0, 5, f"Uploaded: {upload_date}", 0, 1)

                if doc.get('summary'):
                    pdf.set_font('Times', '', 10)
                    pdf.multi_cell(0, 5, f"Summary: {doc['summary'][:200]}..." if len(doc.get('summary', '')) > 200 else f"Summary: {doc['summary']}")

                pdf.ln(3)
        else:
            pdf.set_font('Times', 'I', 10)
            pdf.cell(0, 6, 'No documents uploaded yet.', 0, 1)

        # ==================== TIMELINE ====================
        pdf.add_page()
        pdf.chapter_title('Case Timeline')

        if timeline:
            # Sort by date
            sorted_timeline = sorted(timeline, key=lambda x: x['event_date'], reverse=True)

            for event in sorted_timeline[:20]:  # Limit to 20 events
                event_date = datetime.fromisoformat(event['event_date']).strftime('%d %b %Y')
                event_type = event['event_type'].replace('_', ' ').title()

                pdf.set_font('Times', 'B', 10)
                pdf.cell(40, 6, event_date, 0, 0)
                pdf.set_font('Times', '', 10)
                pdf.cell(0, 6, f"- {event_type}", 0, 1)

                pdf.set_font('Times', 'I', 9)
                pdf.multi_cell(0, 4, f"   {event['description']}")
                pdf.ln(2)
        else:
            pdf.set_font('Times', 'I', 10)
            pdf.cell(0, 6, 'No timeline events yet.', 0, 1)

        # ==================== DEADLINES ====================
        pdf.add_page()
        pdf.chapter_title('Deadlines')

        if deadlines:
            # Pending deadlines
            pending = [d for d in deadlines if not d['is_completed']]
            completed = [d for d in deadlines if d['is_completed']]

            if pending:
                pdf.set_font('Times', 'B', 10)
                pdf.cell(0, 6, 'Upcoming Deadlines:', 0, 1)
                pdf.ln(2)

                for d in sorted(pending, key=lambda x: x['deadline_date']):
                    deadline_date = datetime.fromisoformat(d['deadline_date']).strftime('%d %b %Y')
                    days = d.get('days_until')

                    if days is not None and days <= 3:
                        pdf.set_text_color(255, 0, 0)
                        urgency = "URGENT"
                    elif days is not None and days <= 7:
                        pdf.set_text_color(255, 145, 0)
                        urgency = "Soon"
                    else:
                        pdf.set_text_color(0, 128, 0)
                        urgency = ""

                    pdf.set_font('Times', 'B', 10)
                    pdf.cell(40, 6, deadline_date, 0, 0)
                    pdf.set_font('Times', '', 10)
                    pdf.cell(50, 6, f"{d['deadline_type'].title()}", 0, 0)

                    if urgency:
                        pdf.cell(0, 6, f"({urgency})", 0, 1)
                    else:
                        pdf.cell(0, 6, "", 0, 1)

                    pdf.set_text_color(0, 0, 0)

                    if d.get('description'):
                        pdf.set_font('Times', 'I', 9)
                        pdf.multi_cell(0, 4, f"   {d['description']}")

                    pdf.ln(2)

            if completed:
                pdf.ln(5)
                pdf.set_font('Times', 'B', 10)
                pdf.cell(0, 6, 'Completed Deadlines:', 0, 1)
                pdf.ln(2)

                for d in completed:
                    deadline_date = datetime.fromisoformat(d['deadline_date']).strftime('%d %b %Y')

                    pdf.set_text_color(128, 128, 128)
                    pdf.set_font('Times', '', 10)
                    pdf.cell(40, 6, deadline_date, 0, 0)
                    pdf.cell(50, 6, f"{d['deadline_type'].title()} [COMPLETED]", 0, 1)

                    if d.get('description'):
                        pdf.multi_cell(0, 4, f"   {d['description']}")

                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)
        else:
            pdf.set_font('Times', 'I', 10)
            pdf.cell(0, 6, 'No deadlines set.', 0, 1)

        # ==================== FOOTER NOTE ====================
        pdf.add_page()
        pdf.set_font('Times', 'I', 9)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 5, """
This case summary was generated by LegalAssist AI.

DISCLAIMER: This document is for informational purposes only and does not constitute legal advice. Please consult with a qualified legal professional for advice specific to your situation.

For more information, visit LegalAssist AI or contact your legal representative.
        """)

        # Return PDF as bytes
        out = pdf.output(dest='S')
        return bytes(out) if isinstance(out, bytearray) else out.encode('latin-1')

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return None
    finally:
        db.close()


def generate_anonymized_pdf(case_id: int, anon_id: str) -> Optional[bytes]:
    """
    Generate anonymized PDF for sharing with lawyers.
    Removes personal identifiers.
    """
    db = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return None

        documents = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
        timeline = db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).all()
        deadlines = db.query(CaseDeadline).filter(CaseDeadline.case_id == case_id).all()

        pdf = LegalAssistPDF()
        pdf.add_page()

        # Header
        pdf.set_font('Times', 'B', 16)
        pdf.cell(0, 10, 'Anonymized Case Summary', 0, 1, 'C')
        pdf.set_font('Times', '', 11)
        pdf.cell(0, 8, f"Reference ID: {anon_id}", 0, 1, 'C')
        pdf.ln(5)

        # Case info (anonymized)
        pdf.chapter_title('Case Information')

        info_items = [
            ('Case Type:', case.case_type.title()),
            ('Jurisdiction:', case.jurisdiction),
            ('Status:', case.status.value.title()),
            ('Created:', case.created_at.strftime('%B %Y')),
        ]

        for label, value in info_items:
            pdf.set_font('Times', 'B', 10)
            pdf.cell(40, 6, label, 0, 0)
            pdf.set_font('Times', '', 10)
            pdf.cell(0, 6, value, 0, 1)

        # Documents
        pdf.chapter_title('Documents')

        for i, doc in enumerate(documents):
            pdf.set_font('Times', 'B', 11)
            pdf.cell(0, 6, f"{i+1}. {doc.document_type.value}", 0, 1)

            if doc.summary:
                pdf.set_font('Times', '', 10)
                pdf.multi_cell(0, 5, f"Summary: {doc.summary}")

            pdf.ln(3)

        # Timeline
        pdf.chapter_title('Timeline Events')

        for event in timeline[:15]:
            event_date = event.event_date.strftime('%d %b %Y')
            event_type = event.event_type.replace('_', ' ').title()

            pdf.set_font('Times', 'B', 10)
            pdf.cell(40, 6, event_date, 0, 0)
            pdf.set_font('Times', '', 10)
            pdf.cell(0, 6, f"- {event_type}", 0, 1)

        # Note
        pdf.ln(10)
        pdf.set_font('Times', 'I', 9)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 5, """
This is an ANONYMIZED case summary. Personal identifiers have been removed.
For full case details, please contact the case owner.
        """)

        out = pdf.output(dest='S')
        return bytes(out) if isinstance(out, bytearray) else out.encode('latin-1')

    except Exception as e:
        logger.error(f"Error generating anonymized PDF: {str(e)}")
        return None
    finally:
        db.close()
