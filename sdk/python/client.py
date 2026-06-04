"""
Python Client SDK for Legalassist-AI API
"""
import httpx
from typing import Optional, Dict, Any, List
import asyncio
import time
import uuid


def _with_idempotency_headers(headers: Dict[str, str], idempotency_key: Optional[str] = None) -> Dict[str, str]:
    merged = dict(headers)
    merged["Idempotency-Key"] = idempotency_key or str(uuid.uuid4())
    return merged


class LegalassistClient:
    """Synchronous Python client for Legalassist-AI API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize client
        
        Args:
            base_url: API base URL
            api_key: API key for authentication
            token: JWT token for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
        self.headers = {"User-Agent": "legalassist-python-sdk/1.0.0"}
        
        if api_key:
            self.headers["X-API-Key"] = api_key
        elif token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close HTTP client"""
        self.client.close()
    
    # ========================================================================
    # Authentication
    # ========================================================================
    
    def get_token(self, username: str, password: str) -> Dict[str, Any]:
        """Get access token"""
        response = self.client.post(
            f"{self.base_url}/api/v1/auth/token",
            data={"username": username, "password": password},
            headers=_with_idempotency_headers(self.headers)
        )
        response.raise_for_status()
        return response.json()
    
    def create_api_key(self, name: str, expires_in_days: Optional[int] = None, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Create new API key"""
        payload = {"name": name}
        if expires_in_days:
            payload["expires_in_days"] = expires_in_days
        
        response = self.client.post(
            f"{self.base_url}/api/v1/auth/api-keys",
            json=payload,
            headers=_with_idempotency_headers(self.headers, idempotency_key)
        )
        response.raise_for_status()
        return response.json()
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current user info"""
        response = self.client.get(
            f"{self.base_url}/api/v1/auth/me",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # Document Analysis
    # ========================================================================
    
    def analyze_document(
        self,
        text: Optional[str] = None,
        file_url: Optional[str] = None,
        document_type: str = "unknown",
        extract_remedies: bool = True,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze document asynchronously
        
        Returns job ID to track progress
        """
        payload = {
            "document_type": document_type,
            "extract_remedies": extract_remedies
        }
        
        if text:
            payload["text"] = text
        elif file_url:
            payload["file_url"] = file_url
        else:
            raise ValueError("Must provide text or file_url")
        
        response = self.client.post(
            f"{self.base_url}/api/v1/analyze/document",
            json=payload,
            headers=_with_idempotency_headers(self.headers, idempotency_key)
        )
        response.raise_for_status()
        return response.json()
    
    def get_analysis_status(self, job_id: str) -> Dict[str, Any]:
        """Get analysis job status"""
        response = self.client.get(
            f"{self.base_url}/api/v1/analyze/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_analysis_result(self, job_id: str) -> Dict[str, Any]:
        """Get analysis result"""
        response = self.client.get(
            f"{self.base_url}/api/v1/analyze/{job_id}/result",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_analysis(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 2
    ) -> Dict[str, Any]:
        """Wait for analysis to complete"""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.get_analysis_status(job_id)
            
            if status["status"] == "completed":
                return self.get_analysis_result(job_id)
            elif status["status"] == "failed":
                raise Exception(f"Analysis failed: {status.get('error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Analysis did not complete within {timeout} seconds")
    
    # ========================================================================
    # Case Search
    # ========================================================================
    
    def search_cases(
        self,
        keywords: Optional[List[str]] = None,
        jurisdiction: str = "US",
        limit: int = 10,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for cases"""
        payload = {
            "keywords": keywords or [],
            "jurisdiction": jurisdiction,
            "limit": limit
        }
        
        response = self.client.post(
            f"{self.base_url}/api/v1/cases/search",
            json=payload,
            headers=_with_idempotency_headers(self.headers, idempotency_key)
        )
        response.raise_for_status()
        return response.json()
    
    def get_case_timeline(self, case_id: str) -> Dict[str, Any]:
        """Get case timeline"""
        response = self.client.get(
            f"{self.base_url}/api/v1/cases/{case_id}/timeline",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # Reports
    # ========================================================================
    
    def generate_report(
        self,
        case_id: str,
        report_type: str = "comprehensive",
        format: str = "pdf",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate report asynchronously"""
        payload = {
            "case_id": case_id,
            "report_type": report_type,
            "format": format
        }
        
        response = self.client.post(
            f"{self.base_url}/api/v1/reports/generate",
            json=payload,
            headers=_with_idempotency_headers(self.headers, idempotency_key)
        )
        response.raise_for_status()
        return response.json()
    
    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get report generation status"""
        response = self.client.get(
            f"{self.base_url}/api/v1/reports/{report_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_report(
        self,
        report_id: str,
        timeout: float = 600
    ) -> str:
        """Wait for report and get download URL"""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.get_report_status(report_id)
            
            if status["status"] == "completed":
                return status["download_url"]
            elif status["status"] == "failed":
                raise Exception(f"Report generation failed")
            
            time.sleep(5)
        
        raise TimeoutError(f"Report did not complete within {timeout} seconds")
    
    # ========================================================================
    # Deadlines
    # ========================================================================
    
    def get_upcoming_deadlines(self, days: int = 30) -> Dict[str, Any]:
        """Get upcoming deadlines"""
        response = self.client.get(
            f"{self.base_url}/api/v1/deadlines/upcoming",
            params={"days": days},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def create_deadline(
        self,
        title: str,
        due_date: str,
        description: str = "",
        priority: str = "medium",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new deadline"""
        payload = {
            "title": title,
            "due_date": due_date,
            "description": description,
            "priority": priority
        }
        
        response = self.client.post(
            f"{self.base_url}/api/v1/deadlines",
            json=payload,
            headers=_with_idempotency_headers(self.headers, idempotency_key)
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # Analytics
    # ========================================================================
    
    def get_cost_breakdown(self, period: str = "monthly") -> Dict[str, Any]:
        """Get cost breakdown"""
        response = self.client.get(
            f"{self.base_url}/api/v1/analytics/costs",
            params={"period": period},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_analytics_overview(self) -> Dict[str, Any]:
        """Get analytics overview"""
        response = self.client.get(
            f"{self.base_url}/api/v1/analytics/overview",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()


class AsyncLegalassistClient:
    """Asynchronous Python client for Legalassist-AI API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0
    ):
        """Initialize async client"""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        self.headers = {"User-Agent": "legalassist-python-sdk/1.0.0"}
        
        if api_key:
            self.headers["X-API-Key"] = api_key
        elif token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def analyze_document(
        self,
        text: Optional[str] = None,
        file_url: Optional[str] = None,
        document_type: str = "unknown"
    ) -> Dict[str, Any]:
        """Analyze document asynchronously"""
        payload = {
            "document_type": document_type,
            "text": text or "",
            "file_url": file_url or ""
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/analyze/document",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def get_analysis_result(self, job_id: str) -> Dict[str, Any]:
        """Get analysis result"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/analyze/{job_id}/result",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def wait_for_analysis(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 2
    ) -> Dict[str, Any]:
        """Wait for analysis to complete"""
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                result = await self.get_analysis_result(job_id)
                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 202:
                    # Still processing
                    await asyncio.sleep(poll_interval)
                else:
                    raise
        
        raise TimeoutError(f"Analysis did not complete within {timeout} seconds")


# Example usage
if __name__ == "__main__":
    # Synchronous example
    with LegalassistClient(api_key="your-api-key") as client:
        # Analyze document
        result = client.analyze_document(text="Contract text here...")
        print(f"Job ID: {result['job_id']}")
        
        # Wait for result
        analysis = client.wait_for_analysis(result['job_id'])
        print(f"Summary: {analysis['summary']}")
    
    # Asynchronous example
    async def async_example():
        async with AsyncLegalassistClient(api_key="your-api-key") as client:
            result = await client.analyze_document(text="Contract text...")
            analysis = await client.wait_for_analysis(result['job_id'])
            print(f"Summary: {analysis['summary']}")
    
    # asyncio.run(async_example())
