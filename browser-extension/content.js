// Simple regex to match standard legal citations (e.g. "Roe v. Wade", "Brown v. Board of Education", "347 U.S. 483")
const citationRegex = /\b([A-Z][\w\s]+ v\. [A-Z][\w\s]+|\d+\s+U\.S\.\s+\d+)\b/g;

function highlightPrecedents(node) {
    if (node.nodeType === Node.TEXT_NODE) {
        const text = node.nodeValue;
        if (citationRegex.test(text)) {
            const span = document.createElement('span');
            span.innerHTML = text.replace(citationRegex, '<span class="legalassist-highlight" title="Legal Precedent Detected">$&</span>');
            node.parentNode.replaceChild(span, node);
        }
    } else if (node.nodeType === Node.ELEMENT_NODE && node.nodeName !== 'SCRIPT' && node.nodeName !== 'STYLE' && !node.classList.contains('legalassist-highlight')) {
        for (let i = 0; i < node.childNodes.length; i++) {
            highlightPrecedents(node.childNodes[i]);
        }
    }
}

// Run the highlighter once the page loads
window.addEventListener('load', () => {
    highlightPrecedents(document.body);
    console.log("Legalassist-AI: Precedents highlighted.");
});
