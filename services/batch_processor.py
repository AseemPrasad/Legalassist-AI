import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

logger = logging.getLogger("legalassist.batch")
logger.setLevel(logging.INFO)
# Prevent spamming the root logger
logger.propagate = False

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class BatchPDFProcessor:
    """
    A service for concurrently uploading and processing large batches of PDF documents
    to avoid main-thread blocking and reduce overall ingestion time.
    """
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    def _process_single_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Simulate the extraction and vectorization of a single PDF document.
        In a real scenario, this would call PyPDF2, Tesseract, and an embedding model.
        """
        logger.info(f"Starting processing for: {file_path}")
        
        # Simulate I/O and CPU bound work (e.g. OCR and Embedding)
        time.sleep(1.5) 
        
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 1024 * 500
        
        logger.info(f"Finished processing: {file_path}")
        return {
            "file": file_path,
            "status": "success",
            "extracted_pages": max(1, file_size // (1024 * 50)),
            "vectors_generated": True
        }

    def process_batch(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process a list of PDF file paths concurrently.
        """
        results = []
        failed = []
        start_time = time.time()
        
        logger.info(f"Initiating batch processing for {len(file_paths)} files using {self.max_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(self._process_single_pdf, path): path for path in file_paths}
            
            # As they complete, gather results
            for future in as_completed(future_to_pdf):
                path = future_to_pdf[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    logger.error(f"{path} generated an exception: {exc}")
                    failed.append({"file": path, "error": str(exc)})
                    
        elapsed = time.time() - start_time
        logger.info(f"Batch completed in {elapsed:.2f} seconds. Success: {len(results)}, Failed: {len(failed)}")
        
        return {
            "total_processed": len(file_paths),
            "success_count": len(results),
            "failure_count": len(failed),
            "time_taken_seconds": round(elapsed, 2),
            "failures": failed
        }

# Example Usage
if __name__ == "__main__":
    processor = BatchPDFProcessor(max_workers=5)
    # Simulate a batch of 20 PDFs
    mock_files = [f"/tmp/mock_depositions/deponent_{i}.pdf" for i in range(1, 21)]
    
    summary = processor.process_batch(mock_files)
    print("Batch Summary:", summary)
