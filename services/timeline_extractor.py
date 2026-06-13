import re
from typing import List, Dict, Any
from datetime import datetime

class TimelineExtractor:
    """
    Extracts chronological events and key dates from unstructured legal text
    using regex heuristics.
    """
    
    # Matches common date formats e.g. "January 15, 2020", "15 Jan 2020", "2020-01-15"
    DATE_PATTERN = re.compile(
        r'\b(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}'
        r'|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}'
        r'|\d{4}-\d{2}-\d{2})\b', 
        re.IGNORECASE
    )

    @classmethod
    def extract_events(cls, text: str) -> List[Dict[str, Any]]:
        """
        Parses text for dates and extracts the surrounding sentence as the 'event'.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        events = []
        
        for sentence in sentences:
            match = cls.DATE_PATTERN.search(sentence)
            if match:
                date_str = match.group(0)
                # Try to parse the date for sorting (simplified)
                # For a production system, use dateutil.parser
                events.append({
                    "date_string": date_str,
                    "event_description": sentence.strip(),
                    "confidence": 0.85
                })
                
        return events

# Example Usage
if __name__ == "__main__":
    sample_text = (
        "The contract was originally signed on January 15, 2020. "
        "However, the defendant breached the agreement on 2021-03-12 by failing to deliver the goods. "
        "A formal notice was sent on April 5, 2021. "
        "The plaintiff then filed the lawsuit on 14 Aug 2021."
    )
    
    timeline = TimelineExtractor.extract_events(sample_text)
    for event in timeline:
        print(f"[{event['date_string']}] - {event['event_description']}")
