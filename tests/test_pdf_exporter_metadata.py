from pdf_exporter import LegalAssistPDF


def test_pdf_metadata_settings():
    pdf = LegalAssistPDF()
    pdf.set_title("Test Case Title")
    pdf.set_author("LegalAssist AI Tester")
    pdf.set_creator("Test Export Engine")
    pdf.set_subject("Testing Case Summary Metadata")
    pdf.set_keywords("test, metadata, pdf")
    
    # Assert they are correctly stored on the FPDF metadata attributes
    # In PyFPDF/fpdf2, they are stored under title, author, creator, subject, keywords
    assert getattr(pdf, "str_title", "") == "Test Case Title" or getattr(pdf, "title", "") == "Test Case Title"
    assert getattr(pdf, "str_author", "") == "LegalAssist AI Tester" or getattr(pdf, "author", "") == "LegalAssist AI Tester"
