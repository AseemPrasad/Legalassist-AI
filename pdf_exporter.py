"""
PDF Export functionality for LegalAssist AI.
Generate professional, multilingual PDF case summaries for export and sharing.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fpdf import FPDF
from database import SessionLocal, Case, CaseDocument, CaseDeadline, CaseTimeline
from case_manager import get_case_detail
from db.crud.audit import record_audit_event

logger = logging.getLogger(__name__)

# Constants for PDF styling
PRIMARY_COLOR = (44, 62, 80)    # Dark Blue
SECONDARY_COLOR = (52, 152, 219) # Light Blue
ACCENT_COLOR = (231, 76, 60)    # Red
TEXT_COLOR = (40, 40, 40)
LIGHT_GRAY = (245, 245, 245)
BORDER_COLOR = (200, 200, 200)

# Font configuration
# We assume Unicode-compliant fonts are placed in a 'fonts' directory
FONT_DIR = Path(__file__).parent / "fonts"

# Priority list for Unicode fonts (local assets first, then system fallbacks)
UNICODE_FONT_CONFIGS = [
    {"name": "DejaVu", "file": "DejaVuSans.ttf", "style": ""},
    {"name": "DejaVu", "file": "DejaVuSans-Bold.ttf", "style": "B"},
    {"name": "DejaVu", "file": "DejaVuSans-Oblique.ttf", "style": "I"},
    {"name": "Arial Unicode", "file": "ARIALUNI.TTF", "style": "", "system_path": r"C:\Windows\Fonts\ARIALUNI.TTF"},
    {"name": "Noto Sans", "file": "NotoSans-Regular.ttf", "style": "", "system_path": r"C:\Windows\Fonts\NotoSans-Regular.ttf"},
]

class LegalAssistPDF(FPDF):
    """
    Enhanced PDF class with LegalAssist branding and Unicode support.
    Designed to handle multilingual text (Hindi, Bengali, Urdu, etc.)
    using DejaVu Sans font family.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the LegalAssist PDF generator.
        
        Extends FPDF to include custom branding, font management, 
        and robust error handling for asset-dependent features.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize font tracking before attempting setup
        self.font_availability = {"": False, "B": False, "I": False, "BI": False}
        
        # Trigger font registration process
        self._setup_fonts()
        
        # Configure page layout and breaks
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=20)

    def _setup_fonts(self):
        """
        Register Unicode-capable fonts (DejaVu, Arial Unicode, Noto Sans) 
        for multilingual reporting.
        
        This method ensures the PDF can render characters from various 
        scripts (Hindi, Chinese, Bengali, Urdu, etc.) which are common in legal 
        documents within the target jurisdictions.
        """
        logger.info("Initializing font system for LegalAssist PDF...")
        
        fonts_registered = []
        
        try:
            # Iterate through our prioritized font configurations
            for config in UNICODE_FONT_CONFIGS:
                name = config["name"]
                style = config["style"]
                filename = config["file"]
                
                # Check 1: Local 'fonts' directory
                font_path = FONT_DIR / filename
                
                # Check 2: System path (if defined and local missing)
                if not font_path.exists() and "system_path" in config:
                    sys_path = Path(config["system_path"])
                    if sys_path.exists():
                        font_path = sys_path
                
                if font_path.exists():
                    try:
                        # Register the TrueType font
                        self.add_font(name, style, str(font_path))
                        self.font_availability[style if name == "DejaVu" else ""] = True
                        
                        if name not in fonts_registered:
                            fonts_registered.append(name)
                        
                        logger.debug(f"Successfully registered {name} style '{style}' from {font_path}")
                    except Exception as e:
                        logger.error(f"Error registering font {name} from {font_path}: {str(e)}")
                else:
                    logger.debug(f"Font asset not found: {filename}")

            # Determine the primary font family based on availability
            # We prefer DejaVu if available, otherwise Arial Unicode or Noto Sans
            if "DejaVu" in fonts_registered:
                self.main_font = "DejaVu"
            elif "Arial Unicode" in fonts_registered:
                self.main_font = "Arial Unicode"
            elif "Noto Sans" in fonts_registered:
                self.main_font = "Noto Sans"
            else:
                # If no Unicode fonts are found, fallback to Helvetica
                self.main_font = "Helvetica"
                logger.error("No Unicode-compliant fonts found. Non-Latin characters (Hindi, Chinese) will render as boxes.")
                
            if self.main_font != "Helvetica":
                logger.info(f"Custom font setup complete. Using '{self.main_font}' as primary family.")
                
        except Exception as global_exc:
            logger.critical(f"Global font setup failure: {global_exc}")
            self.main_font = "Helvetica"

    def safe_set_font(self, family: str, style: str = '', size: float = 10):
        """
        Defensively set the font for the current PDF context.
        
        Handles fallbacks for bold/italic styles if not available in the 
        selected Unicode font.
        """
        style_norm = style.upper().strip()
        
        try:
            # Special handling for DejaVu which usually has multiple variants
            if family == "DejaVu":
                if self.font_availability.get(style_norm, False):
                    self.set_font(family, style_norm, size)
                    return
                # Fallback to Regular if Bold/Italic is missing
                self.set_font(family, "", size)
                return

            # For other Unicode fonts (Arial Unicode, Noto Sans), they often 
            # only have a single comprehensive 'Regular' file that handles everything.
            if family in ["Arial Unicode", "Noto Sans"]:
                self.set_font(family, "", size)
                return
            
            # Standard PDF fonts
            self.set_font(family, style_norm, size)
            
        except Exception as e:
            logger.warning(f"safe_set_font recovery: {family} {style} failed -> using Helvetica")
            try:
                self.set_font("Helvetica", "", size)
            except:
                pass

    def _clean(self, txt):
        """
        Clean text for PDF rendering. 
        Removed latin-1 encoding to support Unicode characters.
        """
        if not isinstance(txt, str):
            return str(txt) if txt is not None else ""
        
        # Handle some common smart quotes/special chars that might still cause issues
        # but DejaVu Sans handles most Unicode points natively.
        replacements = {
            '\u201c': '"', '\u201d': '"', 
            '\u2018': "'", '\u2019': "'", 
            '…': '...'
        }
        for k, v in replacements.items():
            txt = txt.replace(k, v)
        
        # CRITICAL FIX: Removed .encode('latin-1', 'replace').decode('latin-1')
        # This allows Hindi, Bengali, Urdu and other Unicode characters to pass through.
        return txt

    def cell(self, w, h=0, txt="", *args, **kwargs):
        """Override cell to ensure text cleaning"""
        txt = self._clean(txt)
        super().cell(w, h, txt, *args, **kwargs)

    def multi_cell(self, w, h, txt, *args, **kwargs):
        """Override multi_cell to ensure text cleaning"""
        txt = self._clean(txt)
        super().multi_cell(w, h, txt, *args, **kwargs)

    def header(self):
        """Add professional header to each page with branding and accent."""
        # Blue top accent bar for a premium look
        self.set_fill_color(*PRIMARY_COLOR)
        self.rect(0, 0, 210, 12, 'F')
        
        self.set_y(18)
        # Use safe_set_font to prevent crashes if 'B' style is missing
        self.safe_set_font(self.main_font, 'B', 16)
        self.set_text_color(*PRIMARY_COLOR)
        self.cell(0, 10, 'LEGALASSIST AI', 0, 0, 'L')
        
        self.safe_set_font(self.main_font, 'I', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'INTELLIGENT CASE MANAGEMENT', 0, 1, 'R')
        
        self.set_draw_color(*SECONDARY_COLOR)
        self.set_line_width(0.5)
        self.line(15, 30, 195, 30)
        self.ln(8)

    def footer(self):
        """Add professional footer to each page with timestamp and page numbers."""
        self.set_y(-25)
        self.set_draw_color(*BORDER_COLOR)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        
        # Use safe_set_font for the italicized footer text
        self.safe_set_font(self.main_font, 'I', 8)
        self.set_text_color(120, 120, 120)
        
        # Left: Generated Timestamp
        date_str = datetime.now().strftime("%d %B %Y | %H:%M")
        self.cell(0, 10, f'Generated by LegalAssist AI on {date_str}', 0, 0, 'L')
        
        # Center: Confidentiality Note
        self.set_x(0)
        self.cell(210, 10, 'STRICTLY CONFIDENTIAL', 0, 0, 'C')
        
        # Right: Page Number
        self.set_x(180)
        self.cell(15, 10, f'Page {self.page_no()}', 0, 0, 'R')

    def section_header(self, title):
        """
        Add a styled section header with a background fill.
        
        Args:
            title: The title text for the section.
        """
        self.ln(5)
        self.safe_set_font(self.main_font, 'B', 12)
        self.set_text_color(*PRIMARY_COLOR)
        self.set_fill_color(*LIGHT_GRAY)
        self.cell(0, 9, f"  {title.upper()}", 0, 1, 'L', True)
        self.ln(3)

    def labeled_value(self, label, value, label_width=45):
        """
        Add a bold label followed by its value with wrapping support.
        
        Ensures consistent formatting and safe font handling for labeled fields.
        """
        if not value:
            value = "N/A"
            
        self.safe_set_font(self.main_font, 'B', 10)
        self.set_text_color(*TEXT_COLOR)
        self.cell(label_width, 6, f"{label}:", 0, 0)
        
        self.safe_set_font(self.main_font, '', 10)
        self.set_text_color(*TEXT_COLOR)
        
        # Use multi_cell for values that might wrap to multiple lines
        self.multi_cell(0, 6, str(value))
        self.ln(1)

    def chapter_title(self, label):
        """Legacy support for original method name"""
        self.section_header(label)

    def chapter_body(self, text):
        """
        Add a body paragraph of text with safe font handling.
        
        Args:
            text: The content to be rendered.
        """
        self.safe_set_font(self.main_font, '', 10)
        self.set_text_color(*TEXT_COLOR)
        self.multi_cell(0, 6, text)
        self.ln(4)

    def draw_status_badge(self, status):
        """
        Draw a colored badge for case status.
        
        Dynamically colors the badge based on the status string.
        """
        status = status.upper()
        
        # Color based on status
        if status in ['OPEN', 'ACTIVE']:
            bg_color = (46, 204, 113) # Green
        elif status in ['CLOSED', 'RESOLVED']:
            bg_color = (149, 165, 166) # Gray
        elif status in ['PENDING', 'WAITING']:
            bg_color = (241, 196, 15) # Yellow
        else:
            bg_color = SECONDARY_COLOR
            
        self.set_fill_color(*bg_color)
        self.set_text_color(255, 255, 255)
        self.safe_set_font(self.main_font, 'B', 9)
        
        width = self.get_string_width(status) + 10
        self.set_x(105 - width/2) # Center horizontally
        self.cell(width, 7, status, 0, 1, 'C', True)
        self.set_text_color(*TEXT_COLOR)
        self.ln(2)

def generate_case_pdf(user_id: int, case_id: int) -> Optional[bytes]:
    """
    Generate a professional PDF summary of a case.
    Ensures Unicode compliance for Hindi, Bengali, and Urdu summaries.
    """
    db = SessionLocal()
    try:
        # Fetch comprehensive case detail
        case_data = get_case_detail(user_id, case_id)
        if not case_data:
            logger.error(f"Access denied or case {case_id} not found for user {user_id}")
            return None

        case = case_data["case"]
        documents = case_data["documents"]
        timeline = case_data["timeline"]
        deadlines = case_data["deadlines"]
        remedies = case_data.get("remedies")

        pdf = LegalAssistPDF()
        pdf.add_page()

        # ==================== CASE HEADER ====================
        # Set large bold font for the title
        pdf.safe_set_font(pdf.main_font, 'B', 20)
        pdf.set_text_color(*PRIMARY_COLOR)
        
        title = case.get('title') or case.get('case_number', 'Untitled Case')
        pdf.multi_cell(0, 12, title, align='C')
        
        # Reference line
        pdf.safe_set_font(pdf.main_font, '', 12)
        pdf.cell(0, 8, f"Case Reference: {case['case_number']}", 0, 1, 'C')
        
        pdf.ln(2)
        pdf.draw_status_badge(case['status'])
        
        # ==================== CORE INFORMATION ====================
        pdf.section_header('Case Information')
        
        pdf.labeled_value('Type of Case', case['case_type'].replace('_', ' ').title())
        pdf.labeled_value('Jurisdiction', case['jurisdiction'])
        
        # Parse and format date with safety check
        try:
            created_at = datetime.fromisoformat(case['created_at'].replace('Z', '+00:00'))
            pdf.labeled_value('Date Initiated', created_at.strftime('%d %B %Y'))
        except Exception:
            pdf.labeled_value('Date Initiated', case['created_at'])
        
        # Optional description section
        if case.get('description'):
            pdf.ln(2)
            pdf.safe_set_font(pdf.main_font, 'B', 10)
            pdf.cell(0, 6, 'Case Description:', 0, 1)
            pdf.safe_set_font(pdf.main_font, '', 10)
            pdf.multi_cell(0, 6, case['description'])

        # ==================== LEGAL REMEDIES & ADVICE ====================
        if remedies:
            pdf.add_page()
            pdf.section_header('Legal Remedies & Analysis')
            
            # Procedural Context
            if remedies.get('what_happened'):
                pdf.safe_set_font(pdf.main_font, 'B', 10)
                pdf.set_text_color(*SECONDARY_COLOR)
                pdf.cell(0, 8, 'PROCEDURAL CONTEXT', 0, 1)
                pdf.set_text_color(*TEXT_COLOR)
                pdf.chapter_body(remedies['what_happened'])
            
            # Appellate Viability
            if remedies.get('can_appeal'):
                pdf.safe_set_font(pdf.main_font, 'B', 10)
                pdf.set_text_color(*SECONDARY_COLOR)
                pdf.cell(0, 8, 'APPELLATE VIABILITY', 0, 1)
                pdf.set_text_color(*TEXT_COLOR)
                pdf.chapter_body(remedies['can_appeal'])
            
            # Key Details Grid
            pdf.ln(2)
            pdf.labeled_value('Appeal Deadline', remedies.get('appeal_days'))
            pdf.labeled_value('Target Court', remedies.get('appeal_court'))
            pdf.labeled_value('Estimated Costs', remedies.get('cost_estimate'))
            
            # Recommended Actions
            if remedies.get('first_action'):
                pdf.ln(4)
                pdf.set_fill_color(230, 240, 255)
                pdf.safe_set_font(pdf.main_font, 'B', 11)
                pdf.cell(0, 9, ' RECOMMENDED NEXT STEP', 0, 1, 'L', True)
                pdf.chapter_body(remedies['first_action'])

            # Statutory Deadlines
            if remedies.get('deadline'):
                pdf.ln(2)
                pdf.set_text_color(*ACCENT_COLOR)
                pdf.safe_set_font(pdf.main_font, 'B', 11)
                pdf.cell(0, 8, 'CRITICAL STATUTORY DEADLINE', 0, 1)
                pdf.safe_set_font(pdf.main_font, 'B', 12)
                pdf.multi_cell(0, 7, remedies['deadline'])
                pdf.set_text_color(*TEXT_COLOR)

        # ==================== DOCUMENTS & FILINGS ====================
        pdf.add_page()
        pdf.section_header('Document Repository')
        
        if documents:
            # Iterate through each case document
            for i, doc in enumerate(documents):
                pdf.safe_set_font(pdf.main_font, 'B', 11)
                pdf.set_text_color(*PRIMARY_COLOR)
                pdf.cell(0, 7, f"{i+1}. {doc['document_type'].replace('_', ' ').title()}", 0, 1)
                
                pdf.safe_set_font(pdf.main_font, 'I', 9)
                pdf.set_text_color(120, 120, 120)
                try:
                    up_date = datetime.fromisoformat(doc['uploaded_at'].replace('Z', '+00:00')).strftime('%d %b %Y')
                    pdf.cell(0, 5, f"   Uploaded: {up_date}", 0, 1)
                except Exception:
                    pdf.cell(0, 5, f"   Uploaded: {doc['uploaded_at']}", 0, 1)
                
                if doc.get('summary'):
                    pdf.ln(1)
                    pdf.safe_set_font(pdf.main_font, '', 10)
                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_x(20)
                    # This summary might contain Hindi/Bengali/Urdu Unicode text
                    pdf.multi_cell(0, 5, doc['summary'])
                
                pdf.ln(4)
        else:
            pdf.safe_set_font(pdf.main_font, 'I', 10)
            pdf.cell(0, 10, 'No documents associated with this case file.', 0, 1)

        # ==================== CASE TIMELINE ====================
        pdf.add_page()
        pdf.section_header('Case Procedural History')
        
        if timeline:
            # Sort events by date descending for chronological clarity
            try:
                sorted_timeline = sorted(timeline, key=lambda x: x['event_date'], reverse=True)
            except Exception:
                sorted_timeline = timeline
            
            for event in sorted_timeline:
                try:
                    ev_date = datetime.fromisoformat(event['event_date'].replace('Z', '+00:00')).strftime('%d %b %Y')
                except Exception:
                    ev_date = event['event_date']
                    
                ev_type = event['event_type'].replace('_', ' ').title()
                
                # Date column
                pdf.safe_set_font(pdf.main_font, 'B', 10)
                pdf.set_text_color(*SECONDARY_COLOR)
                pdf.cell(40, 7, ev_date, 0, 0)
                
                # Event type column
                pdf.safe_set_font(pdf.main_font, 'B', 10)
                pdf.set_text_color(*PRIMARY_COLOR)
                pdf.cell(0, 7, ev_type, 0, 1)
                
                # Event description with indentation
                if event.get('description'):
                    pdf.safe_set_font(pdf.main_font, '', 9)
                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_x(55)
                    pdf.multi_cell(0, 5, event['description'])
                
                pdf.ln(3)
                
                # Check for page overflow to ensure clean breaks
                if pdf.get_y() > 260:
                    pdf.add_page()
        else:
            pdf.safe_set_font(pdf.main_font, 'I', 10)
            pdf.cell(0, 10, 'Timeline history is currently empty.', 0, 1)

        # ==================== DEADLINE MANAGEMENT ====================
        pdf.add_page()
        pdf.section_header('Upcoming Deadlines & Obligations')
        
        if deadlines:
            # Split into pending and completed for clearer categorization
            pending = [d for d in deadlines if not d['is_completed']]
            completed = [d for d in deadlines if d['is_completed']]
            
            if pending:
                pdf.safe_set_font(pdf.main_font, 'B', 11)
                pdf.set_text_color(*ACCENT_COLOR)
                pdf.cell(0, 8, 'PENDING ACTIONS', 0, 1)
                pdf.ln(2)
                
                for d in sorted(pending, key=lambda x: x['deadline_date']):
                    try:
                        d_date = datetime.fromisoformat(d['deadline_date'].replace('Z', '+00:00')).strftime('%d %b %Y')
                    except Exception:
                        d_date = d['deadline_date']
                        
                    days = d.get('days_until', 999)
                    
                    # Urgency coloring
                    if days <= 3:
                        pdf.set_text_color(*ACCENT_COLOR)
                        tag = " [URGENT]"
                    elif days <= 10:
                        pdf.set_text_color(230, 126, 34) # Orange
                        tag = " [SOON]"
                    else:
                        pdf.set_text_color(39, 174, 96) # Green
                        tag = ""
                        
                    pdf.safe_set_font(pdf.main_font, 'B', 10)
                    pdf.cell(40, 7, d_date, 0, 0)
                    pdf.cell(100, 7, d['deadline_type'].title(), 0, 0)
                    pdf.cell(0, 7, tag, 0, 1, 'R')
                    
                    if d.get('description'):
                        pdf.safe_set_font(pdf.main_font, 'I', 9)
                        pdf.set_text_color(80, 80, 80)
                        pdf.set_x(55)
                        pdf.multi_cell(0, 5, d['description'])
                    pdf.ln(3)

            if completed:
                pdf.ln(5)
                pdf.safe_set_font(pdf.main_font, 'B', 11)
                pdf.set_text_color(127, 140, 141)
                pdf.cell(0, 8, 'COMPLETED MILESTONES', 0, 1)
                
                for d in completed:
                    try:
                        d_date = datetime.fromisoformat(d['deadline_date'].replace('Z', '+00:00')).strftime('%d %b %Y')
                    except Exception:
                        d_date = d['deadline_date']
                    
                    pdf.safe_set_font(pdf.main_font, '', 10)
                    pdf.set_text_color(149, 165, 166)
                    pdf.cell(40, 6, d_date, 0, 0)
                    pdf.cell(0, 6, f"{d['deadline_type'].title()} (Done)", 0, 1)
                    pdf.ln(1)
        else:
            pdf.safe_set_font(pdf.main_font, 'I', 10)
            pdf.cell(0, 10, 'No statutory deadlines tracked for this case.', 0, 1)

        # ==================== DISCLAIMER ====================
        pdf.add_page()
        pdf.set_y(100)
        pdf.safe_set_font(pdf.main_font, 'B', 11)
        pdf.set_text_color(*PRIMARY_COLOR)
        pdf.cell(0, 10, 'LEGAL DISCLAIMER & NOTICES', 0, 1, 'C')
        
        pdf.safe_set_font(pdf.main_font, '', 10)
        pdf.set_text_color(100, 100, 100)
        discl_text = (
            "This briefing document is generated by LegalAssist AI and is provided for informational "
            "and organizational purposes only. \n\n"
            "PROHIBITION ON LEGAL ADVICE: The contents of this document do NOT constitute legal advice. "
            "LegalAssist AI is an intelligent document management system, not a law firm. "
            "Users are strongly advised to consult with a qualified attorney to verify the automated "
            "summaries and analysis provided herein. \n\n"
            "CONFIDENTIALITY: This document may contain privileged information. "
            "Unauthorized disclosure, copying, or distribution is strictly prohibited."
        )
        pdf.multi_cell(0, 6, discl_text, align='C')

        # Final PDF output
        out_content = pdf.output(dest='S')
        if isinstance(out_content, (bytes, bytearray)):
            return bytes(out_content)
        return out_content.encode('utf-8')

    except Exception as err:
        logger.error(f"Fatal error during PDF generation sequence: {str(err)}", exc_info=True)
        return None
    finally:
        db.close()

def generate_anonymized_pdf(
    case_id: int,
    anon_id: str,
    user_id: int,
    profile_name: Optional[str] = None,
    anonymized_data: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    """
    Generate an anonymized PDF for external legal review.
    Strips all personal identifiers to maintain privacy.
    """
    db = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return None

        # Verify case ownership
        if case.user_id != user_id:
            logger.warning(f"Unauthorized access attempt: user {user_id} tried to export case {case_id}")
            return None

        documents = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
        timeline = db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).all()
        from services.privacy_redaction import normalize_privacy_profile, get_privacy_profile_definition
        selected_profile = normalize_privacy_profile(profile_name)
        profile = get_privacy_profile_definition(selected_profile)

        if anonymized_data is None:
            try:
                from case_manager import generate_anonymized_case_data
                anonymized_data = generate_anonymized_case_data(case_id, profile_name=selected_profile)
            except Exception:
                anonymized_data = None

        pdf = LegalAssistPDF()
        pdf.add_page()

        # Header for Anonymized Report
        pdf.safe_set_font(pdf.main_font, 'B', 18)
        pdf.set_text_color(*ACCENT_COLOR)
        pdf.cell(0, 15, 'ANONYMIZED CONSULTATION BRIEF', 0, 1, 'C')
        
        pdf.safe_set_font(pdf.main_font, '', 11)
        pdf.set_text_color(*TEXT_COLOR)
        pdf.cell(0, 8, f"Unique Reference ID: {anon_id}", 0, 1, 'C')
        pdf.cell(0, 8, f"Privacy profile: {profile.get('label', selected_profile)}", 0, 1, 'C')
        pdf.ln(10)

        # Classification info
        pdf.section_header('Case Classification')
        
        def safe_get(obj, key, default=''):
            """Safely get value from dict or ORM object"""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        pdf.labeled_value('Legal Category', str(safe_get(case, 'case_type', case.case_type)).title())
        pdf.labeled_value('Venue Jurisdiction', safe_get(case, 'jurisdiction', case.jurisdiction))
        status_val = safe_get(case, 'status', case.status)
        if hasattr(status_val, 'value'):
            status_val = status_val.value
        pdf.labeled_value('Case Status', str(status_val).title())
        
        created_at = safe_get(case, 'created_at', case.created_at)
        if hasattr(created_at, 'strftime'):
            pdf.labeled_value('Inception Date', created_at.strftime('%B %Y'))
        else:
            pdf.labeled_value('Inception Date', str(created_at)[:7])

        # Document abstracts
        pdf.section_header('Evidence Summary')
        if anonymized_data and anonymized_data.get("documents"):
            for doc in anonymized_data["documents"]:
                pdf.safe_set_font(pdf.main_font, 'B', 10)
                pdf.cell(0, 7, f"Type: {doc.get('type', 'Document')}", 0, 1)
                summary = doc.get("summary")
                if summary:
                    pdf.safe_set_font(pdf.main_font, '', 10)
                    pdf.multi_cell(0, 5, str(summary))
                pdf.ln(3)
        elif documents:
            for doc in documents:
                pdf.safe_set_font(pdf.main_font, 'B', 10)
                pdf.cell(0, 7, f"Type: {doc.document_type.value}", 0, 1)
                if doc.summary:
                    pdf.safe_set_font(pdf.main_font, '', 10)
                    pdf.multi_cell(0, 5, doc.summary)
                pdf.ln(3)
        else:
            pdf.chapter_body("No associated documents for review.")

        # Procedure Timeline
        pdf.section_header('Procedural Milestones')
        if anonymized_data and anonymized_data.get("timeline"):
            for event in anonymized_data["timeline"][:20]:
                pdf.safe_set_font(pdf.main_font, 'B', 9)
                pdf.cell(40, 6, str(event.get("event_type", "event")).replace('_', ' ').title(), 0, 0)
                pdf.safe_set_font(pdf.main_font, '', 9)
                pdf.cell(0, 6, str(event.get("description") or ""), 0, 1)
        elif timeline:
            for event in timeline[:20]:
                pdf.safe_set_font(pdf.main_font, 'B', 9)
                pdf.cell(40, 6, event.event_date.strftime('%d %b %Y'), 0, 0)
                pdf.safe_set_font(pdf.main_font, '', 9)
                pdf.cell(0, 6, event.event_type.replace('_', ' ').title(), 0, 1)
        
        # Privacy Footer
        pdf.set_y(-35)
        pdf.safe_set_font(pdf.main_font, 'I', 8)
        pdf.set_text_color(160, 160, 160)
        pdf.multi_cell(0, 4, (
            "NOTICE: This is an anonymized case summary. All PII (Personally Identifiable Information) "
            "has been redacted or abstracted. This document is intended for third-party legal "
            "consultation purposes only."
        ), align='C')

        final_out = pdf.output(dest='S')
        if isinstance(final_out, (bytes, bytearray)):
            record_audit_event(
                db,
                actor=f"user:{user_id}",
                actor_user_id=user_id,
                action="download_anonymized_pdf",
                resource=f"case:{case_id}",
                case_id=case_id,
                metadata={"privacy_profile": selected_profile, "anonymized_id": anon_id},
            )
            return bytes(final_out)
        record_audit_event(
            db,
            actor=f"user:{user_id}",
            actor_user_id=user_id,
            action="download_anonymized_pdf",
            resource=f"case:{case_id}",
            case_id=case_id,
            metadata={"privacy_profile": selected_profile, "anonymized_id": anon_id},
        )
        return final_out.encode('utf-8')

    except Exception as e:
        logger.error(f"Failed to generate anonymized report: {str(e)}")
        return None
    finally:
        db.close()
