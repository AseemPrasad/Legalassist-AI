from core import extract_text_from_pdf
from core.app_utils import PDFProcessingError
import io

b = b"%PDF-1.4\n1 0 obj<< /Type /Catalog >>\n/JavaScript (alert)\n%%EOF"
try:
    extract_text_from_pdf(io.BytesIO(b), enable_ocr=False)
    print('NO_ERROR')
except PDFProcessingError as e:
    print('RAISED_PDFPROCESSINGERROR', str(e))
except Exception as e:
    print('OTHER_ERROR', type(e), e)
