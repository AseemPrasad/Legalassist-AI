"""Schema-driven case export builder.

This module centralizes export field selection and serializes the same payload
into JSON, PDF, DOCX, or ZIP bundle outputs.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import xml.sax.saxutils as saxutils
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

from fpdf import FPDF
from sqlalchemy.orm import Session

from db.models import Attachment, Case, CaseDeadline, CaseDocument, CaseTimeline
from database import SessionLocal
from db.crud.audit import record_audit_event
from services.privacy_redaction import (
    apply_privacy_profile,
    normalize_privacy_profile,
)


ExportPayload = Dict[str, Any]


@dataclass(frozen=True)
class ExportFieldDefinition:
    field_id: str
    label: str
    description: str
    group: str
    default_selected: bool = False


@dataclass(frozen=True)
class ExportArtifact:
    file_name: str
    mime_type: str
    format: str
    data: bytes
    case_ids: List[int]
    selected_fields: List[str]


FIELD_SCHEMA: List[ExportFieldDefinition] = [
    ExportFieldDefinition("case_number", "Case number", "Internal case reference.", "case", True),
    ExportFieldDefinition("title", "Title", "Display title for the matter.", "case", True),
    ExportFieldDefinition("case_type", "Case type", "Case classification.", "case", True),
    ExportFieldDefinition("jurisdiction", "Jurisdiction", "Forum or venue.", "case", False),
    ExportFieldDefinition("status", "Status", "Current lifecycle status.", "case", True),
    ExportFieldDefinition("created_at", "Created at", "Creation timestamp.", "case", True),
    ExportFieldDefinition("updated_at", "Updated at", "Most recent update timestamp.", "case", False),
    ExportFieldDefinition("document_count", "Document count", "Total number of documents.", "summary", True),
    ExportFieldDefinition("latest_document", "Latest document", "Most recently uploaded document.", "summary", True),
    ExportFieldDefinition("next_deadline", "Next deadline", "Earliest open upcoming deadline.", "summary", True),
    ExportFieldDefinition("documents", "Documents", "All documents attached to the case.", "sections", False),
    ExportFieldDefinition("deadlines", "Deadlines", "All deadlines for the case.", "sections", True),
    ExportFieldDefinition("timeline", "Timeline", "Case timeline events.", "sections", True),
    ExportFieldDefinition("attachments", "Attachments", "Uploaded attachments.", "sections", False),
    ExportFieldDefinition("remedies", "Remedies", "Structured remedies data from the latest document.", "sections", False),
]

SUPPORTED_FORMATS = ("json", "pdf", "docx")
DEFAULT_SELECTED_FIELDS = [
    field.field_id for field in FIELD_SCHEMA if field.default_selected
]


def get_export_field_options() -> List[Dict[str, Any]]:
    return [
        {
            "field_id": field.field_id,
            "label": field.label,
            "description": field.description,
            "group": field.group,
            "default_selected": field.default_selected,
        }
        for field in FIELD_SCHEMA
    ]


def get_default_export_fields() -> List[str]:
    return list(DEFAULT_SELECTED_FIELDS)


def _safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name or "export")
    name = name.strip(" .") or "export"
    return name[:180]


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _normalize_fields(field_ids: Optional[Sequence[str]]) -> List[str]:
    requested = [field for field in (field_ids or []) if field in {item.field_id for item in FIELD_SCHEMA}]
    if requested:
        seen = set()
        ordered: List[str] = []
        for field_id in requested:
            if field_id not in seen:
                ordered.append(field_id)
                seen.add(field_id)
        return ordered
    return get_default_export_fields()


def _serialize_case(case: Case, field_ids: Sequence[str]) -> Dict[str, Any]:
    selected = set(field_ids)
    payload: Dict[str, Any] = {}

    if "case_number" in selected:
        payload["case_number"] = case.case_number
    if "title" in selected:
        payload["title"] = case.title or case.case_number
    if "case_type" in selected:
        payload["case_type"] = case.case_type
    if "jurisdiction" in selected:
        payload["jurisdiction"] = case.jurisdiction
    if "status" in selected:
        payload["status"] = case.status.value if hasattr(case.status, "value") else str(case.status)
    if "created_at" in selected:
        payload["created_at"] = _iso(case.created_at)
    if "updated_at" in selected:
        payload["updated_at"] = _iso(case.updated_at)

    return payload


def _serialize_document(doc: CaseDocument) -> Dict[str, Any]:
    return {
        "id": doc.id,
        "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else str(doc.document_type),
        "uploaded_at": _iso(doc.uploaded_at),
        "summary": doc.summary,
        "has_remedies": bool(doc.remedies),
        "source_attachment_id": doc.source_attachment_id,
        "extracted_metadata": doc.extracted_metadata,
        "extraction_method": doc.extraction_method,
        "ocr_used": bool(doc.ocr_used),
    }


def _serialize_deadline(deadline: CaseDeadline) -> Dict[str, Any]:
    return {
        "id": deadline.id,
        "deadline_type": deadline.deadline_type,
        "deadline_date": _iso(deadline.deadline_date),
        "description": deadline.description,
        "is_completed": bool(deadline.is_completed),
        "days_until": deadline.days_until_deadline(),
    }


def _serialize_timeline(event: CaseTimeline) -> Dict[str, Any]:
    return {
        "id": event.id,
        "event_date": _iso(event.event_date),
        "event_type": event.event_type,
        "description": event.description,
        "metadata": event.event_metadata or {},
    }


def _serialize_attachment(attachment: Attachment) -> Dict[str, Any]:
    return {
        "id": attachment.id,
        "original_filename": attachment.original_filename,
        "uploaded_at": _iso(attachment.uploaded_at),
        "size_bytes": attachment.size_bytes,
        "content_type": attachment.content_type,
        "document_id": attachment.document_id,
    }


def _serialize_next_deadline(deadlines: Sequence[CaseDeadline]) -> Optional[Dict[str, Any]]:
    upcoming = [
        deadline
        for deadline in deadlines
        if not deadline.is_completed and deadline.deadline_date and deadline.deadline_date > dt.datetime.now(dt.timezone.utc)
    ]
    if not upcoming:
        return None
    next_deadline = sorted(upcoming, key=lambda item: item.deadline_date)[0]
    return _serialize_deadline(next_deadline)


def _load_case_payload(db: Session, user_id: int, case_id: int) -> Optional[Dict[str, Any]]:
    case = db.query(Case).filter(Case.id == case_id, Case.user_id == user_id).first()
    if not case:
        return None

    documents = (
        db.query(CaseDocument)
        .filter(CaseDocument.case_id == case_id)
        .order_by(CaseDocument.uploaded_at.desc(), CaseDocument.id.desc())
        .all()
    )
    deadlines = (
        db.query(CaseDeadline)
        .filter(CaseDeadline.case_id == case_id)
        .order_by(CaseDeadline.deadline_date.asc(), CaseDeadline.id.asc())
        .all()
    )
    timeline = (
        db.query(CaseTimeline)
        .filter(CaseTimeline.case_id == case_id)
        .order_by(CaseTimeline.event_date.desc(), CaseTimeline.id.desc())
        .all()
    )
    attachments = (
        db.query(Attachment)
        .filter(Attachment.case_id == case_id)
        .order_by(Attachment.uploaded_at.desc(), Attachment.id.desc())
        .all()
    )

    latest_document = documents[0] if documents else None
    next_deadline = _serialize_next_deadline(deadlines)

    return {
        "case": case,
        "documents": documents,
        "deadlines": deadlines,
        "timeline": timeline,
        "attachments": attachments,
        "latest_document": latest_document,
        "next_deadline": next_deadline,
    }


def build_case_export_payload(
    *,
    user_id: int,
    case_id: int,
    field_ids: Optional[Sequence[str]] = None,
    privacy_profile: Optional[str] = None,
    db: Optional[Session] = None,
) -> Optional[Dict[str, Any]]:
    selected_fields = _normalize_fields(field_ids)

    def _run(db: Session) -> Optional[Dict[str, Any]]:
        source = _load_case_payload(db, user_id, case_id)
        if not source:
            return None

        case: Case = source["case"]
        documents: List[CaseDocument] = source["documents"]
        deadlines: List[CaseDeadline] = source["deadlines"]
        timeline: List[CaseTimeline] = source["timeline"]
        attachments: List[Attachment] = source["attachments"]
        latest_document: Optional[CaseDocument] = source["latest_document"]
        next_deadline = source["next_deadline"]

        payload: Dict[str, Any] = {
            "export": {
                "case_id": case.id,
                "case_number": case.case_number,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "selected_fields": list(selected_fields),
            }
        }

        case_payload = _serialize_case(case, selected_fields)
        if case_payload:
            payload["case"] = case_payload

        if "document_count" in selected_fields:
            payload["document_count"] = len(documents)
        if "latest_document" in selected_fields and latest_document:
            payload["latest_document"] = _serialize_document(latest_document)
        if "next_deadline" in selected_fields:
            payload["next_deadline"] = next_deadline
        if "documents" in selected_fields:
            payload["documents"] = [_serialize_document(doc) for doc in documents]
        if "deadlines" in selected_fields:
            payload["deadlines"] = [_serialize_deadline(deadline) for deadline in deadlines]
        if "timeline" in selected_fields:
            payload["timeline"] = [_serialize_timeline(event) for event in timeline]
        if "attachments" in selected_fields:
            payload["attachments"] = [_serialize_attachment(attachment) for attachment in attachments]
        if "remedies" in selected_fields:
            payload["remedies"] = latest_document.remedies if latest_document else None

        selected_profile = normalize_privacy_profile(privacy_profile)
        redacted_payload = apply_privacy_profile(payload, selected_profile)
        record_audit_event(
            db,
            actor=f"user:{user_id}",
            actor_user_id=user_id,
            action="export_preview",
            resource=f"case:{case_id}",
            case_id=case_id,
            metadata={"privacy_profile": selected_profile, "fields": list(selected_fields)},
        )
        return redacted_payload

    if db is not None:
        return _run(db)
    with SessionLocal() as _db:
        return _run(_db)


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _pdf_write_multiline(pdf: FPDF, label: str, value: Any) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 7, f"{label}:")
    pdf.set_font("Helvetica", "", 11)
    if isinstance(value, list):
        if not value:
            pdf.multi_cell(0, 7, "  - None")
        for item in value:
            if isinstance(item, dict):
                text = "; ".join(f"{key}={item[key]}" for key in item.keys())
            else:
                text = str(item)
            pdf.multi_cell(0, 6, f"  - {text}")
    elif isinstance(value, dict):
        if not value:
            pdf.multi_cell(0, 7, "  - None")
        else:
            for key in value.keys():
                pdf.multi_cell(0, 6, f"  - {key}: {value[key]}")
    else:
        pdf.multi_cell(0, 7, f"  {value if value is not None else 'None'}")
    pdf.ln(1)


def _pdf_bytes(payload: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    title = payload.get("case", {}).get("case_number") or payload.get("export", {}).get("case_number") or "Case Export"
    pdf.multi_cell(0, 10, f"LegalAssist AI Export - {title}")
    pdf.ln(2)

    export_meta = payload.get("export", {})
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, f"Generated at: {export_meta.get('generated_at', '')}")
    pdf.multi_cell(0, 6, f"Selected fields: {', '.join(export_meta.get('selected_fields', []))}")
    pdf.ln(2)

    for key, value in payload.items():
        if key == "export":
            continue
        _pdf_write_multiline(pdf, key.replace("_", " ").title(), value)

    out_content = pdf.output(dest="S")
    if isinstance(out_content, (bytes, bytearray)):
        return bytes(out_content)
    return out_content.encode("utf-8")


def _docx_bytes(payload: Dict[str, Any]) -> bytes:
    title = payload.get("case", {}).get("case_number") or payload.get("export", {}).get("case_number") or "Case Export"
    generated_at = payload.get("export", {}).get("generated_at", "")
    parts: List[str] = [
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>",
        "<w:body>",
    ]

    def paragraph(text: str, bold: bool = False) -> str:
        escaped = saxutils.escape(text)
        if bold:
            return (
                "<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>"
                + escaped
                + "</w:t></w:r></w:p>"
            )
        return f"<w:p><w:r><w:t>{escaped}</w:t></w:r></w:p>"

    parts.append(paragraph(f"LegalAssist AI Export - {title}", bold=True))
    parts.append(paragraph(f"Generated at: {generated_at}"))
    parts.append(paragraph(f"Selected fields: {', '.join(payload.get('export', {}).get('selected_fields', []))}"))

    for key, value in payload.items():
        if key == "export":
            continue
        parts.append(paragraph(key.replace("_", " ").title(), bold=True))
        if isinstance(value, list):
            if not value:
                parts.append(paragraph("- None"))
            for item in value:
                if isinstance(item, dict):
                    text = "; ".join(f"{field}={item[field]}" for field in item.keys())
                else:
                    text = str(item)
                parts.append(paragraph(f"- {text}"))
        elif isinstance(value, dict):
            if not value:
                parts.append(paragraph("- None"))
            else:
                for field in value.keys():
                    parts.append(paragraph(f"- {field}: {value[field]}"))
        else:
            parts.append(paragraph(f"- {value if value is not None else 'None'}"))

    parts.append("<w:sectPr><w:pgSz w:w='12240' w:h='15840'/><w:pgMar w:top='1440' w:right='1440' w:bottom='1440' w:left='1440'/></w:sectPr>")
    parts.append("</w:body></w:document>")

    document_xml = "".join(parts)
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
  <Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
  <Default Extension='xml' ContentType='application/xml'/>
  <Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>
</Types>
""")
        archive.writestr("_rels/.rels", """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>
</Relationships>
""")
        archive.writestr("word/document.xml", document_xml)
        archive.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'/>
""",
        )
    return buffer.getvalue()


def build_case_export_artifact(
    *,
    user_id: int,
    case_id: int,
    format: str,
    field_ids: Optional[Sequence[str]] = None,
    privacy_profile: Optional[str] = None,
    db: Optional[Session] = None,
) -> Optional[ExportArtifact]:
    selected_fields = _normalize_fields(field_ids)
    payload = build_case_export_payload(
        user_id=user_id,
        case_id=case_id,
        field_ids=selected_fields,
        privacy_profile=privacy_profile,
        db=db,
    )
    if not payload:
        return None

    format_name = (format or "json").lower()
    if format_name not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {format}")

    case_number = payload.get("export", {}).get("case_number") or f"case-{case_id}"
    base_name = _safe_filename(f"{case_number}_export")

    if format_name == "json":
        data = _json_bytes(payload)
        mime_type = "application/json"
        file_name = f"case_{case_id}_export.json"
    elif format_name == "pdf":
        data = _pdf_bytes(payload)
        mime_type = "application/pdf"
        file_name = f"case_{case_id}_export.pdf"
    else:
        data = _docx_bytes(payload)
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        file_name = f"case_{case_id}_export.docx"

    def _audit(db: Session) -> None:
        record_audit_event(
            db,
            actor=f"user:{user_id}",
            actor_user_id=user_id,
            action="export_download",
            resource=f"case:{case_id}",
            case_id=case_id,
            metadata={
                "format": format_name,
                "privacy_profile": normalize_privacy_profile(privacy_profile),
                "fields": list(selected_fields),
            },
        )

    if db is not None:
        _audit(db)
    else:
        with SessionLocal() as _db:
            _audit(_db)

    return ExportArtifact(
        file_name=file_name,
        mime_type=mime_type,
        format=format_name,
        data=data,
        case_ids=[case_id],
        selected_fields=list(selected_fields),
    )


def build_case_export_bundle(
    *,
    user_id: int,
    case_ids: Sequence[int],
    field_ids: Optional[Sequence[str]] = None,
    formats: Sequence[str] = ("json", "pdf", "docx"),
    privacy_profile: Optional[str] = None,
) -> Optional[ExportArtifact]:
    unique_case_ids = []
    seen_case_ids = set()
    for case_id in case_ids:
        int_case_id = int(case_id)
        if int_case_id not in seen_case_ids:
            unique_case_ids.append(int_case_id)
            seen_case_ids.add(int_case_id)

    if not unique_case_ids:
        return None

    selected_fields = _normalize_fields(field_ids)
    normalized_formats = []
    seen_formats = set()
    for fmt in formats:
        lower = (fmt or "json").lower()
        if lower in SUPPORTED_FORMATS and lower not in seen_formats:
            normalized_formats.append(lower)
            seen_formats.add(lower)
    if not normalized_formats:
        normalized_formats = ["json"]

    buffer = BytesIO()
    manifest: Dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_ids": list(unique_case_ids),
        "formats": list(normalized_formats),
        "fields": list(selected_fields),
        "files": [],
    }

    case_label = "_".join(str(case_id) for case_id in unique_case_ids)

    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        with SessionLocal() as db:
            for case_id in unique_case_ids:
                for format_name in normalized_formats:
                    artifact = build_case_export_artifact(
                        user_id=user_id,
                        case_id=case_id,
                        format=format_name,
                        field_ids=selected_fields,
                        privacy_profile=privacy_profile,
                        db=db,
                    )
                    if not artifact:
                        continue
                    case_folder = f"case_{case_id}"
                    entry_name = f"{case_folder}/{artifact.file_name}"
                    archive.writestr(entry_name, artifact.data)
                    manifest["files"].append(
                        {
                            "case_id": case_id,
                            "format": format_name,
                            "entry": entry_name,
                            "size_bytes": len(artifact.data),
                        }
                    )
                    record_audit_event(
                        db,
                        actor=f"user:{user_id}",
                        actor_user_id=user_id,
                        action="export_bundle_item",
                        resource=f"case:{case_id}",
                        case_id=case_id,
                        metadata={
                            "format": format_name,
                            "privacy_profile": normalize_privacy_profile(privacy_profile),
                            "bundle_entry": entry_name,
                        },
                    )
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

            record_audit_event(
                db,
                actor=f"user:{user_id}",
                actor_user_id=user_id,
                action="export_bundle",
                resource=f"case_bundle:{case_label}",
                metadata={
                    "case_ids": list(unique_case_ids),
                    "formats": list(normalized_formats),
                    "privacy_profile": normalize_privacy_profile(privacy_profile),
                    "fields": list(selected_fields),
                },
            )

    file_name = _safe_filename(f"case_export_bundle_{case_label}.zip")
    return ExportArtifact(
        file_name=file_name,
        mime_type="application/zip",
        format="zip",
        data=buffer.getvalue(),
        case_ids=unique_case_ids,
        selected_fields=list(selected_fields),
    )


def get_case_export_preview(
    *,
    user_id: int,
    case_id: int,
    field_ids: Optional[Sequence[str]] = None,
    privacy_profile: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return build_case_export_payload(
        user_id=user_id,
        case_id=case_id,
        field_ids=field_ids,
        privacy_profile=privacy_profile,
    )
