"""
JavaScript/TypeScript Client SDK for Legalassist-AI API
"""

function generateIdempotencyKey() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `idemp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}


class LegalassistClient {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || "http://localhost:8000";
    this.apiKey = options.apiKey;
    this.token = options.token;
    this.timeout = options.timeout || 30000;
  }

  /**
   * Make HTTP request
   */
  async request(method, path, options = {}) {
    const url = `${this.baseUrl}${path}`;
    const headers = {
      "Content-Type": "application/json",
      "User-Agent": "legalassist-js-sdk/1.0.0",
      ...options.headers,
    };

    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    } else if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
      headers["Idempotency-Key"] = options.idempotencyKey || generateIdempotencyKey();
    }

    const config = {
      method,
      headers,
      timeout: this.timeout,
      ...options,
    };

    const response = await fetch(url, config);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // ========================================================================
  // Authentication
  // ========================================================================

  async getToken(username, password) {
    return this.request("POST", "/api/v1/auth/token", {
      body: JSON.stringify({ username, password }),
      idempotencyKey: generateIdempotencyKey(),
    });
  }

  async createApiKey(name, expiresInDays = null, idempotencyKey = null) {
    return this.request("POST", "/api/v1/auth/api-keys", {
      body: JSON.stringify({
        name,
        expires_in_days: expiresInDays,
      }),
      idempotencyKey,
    });
  }

  async getCurrentUser() {
    return this.request("GET", "/api/v1/auth/me");
  }

  // ========================================================================
  // Document Analysis
  // ========================================================================

  async analyzeDocument(options = {}) {
    const { text, fileUrl, documentType = "unknown" } = options;

    if (!text && !fileUrl) {
      throw new Error("Must provide text or fileUrl");
    }

    const payload = { document_type: documentType };
    if (text) payload.text = text;
    if (fileUrl) payload.file_url = fileUrl;

    return this.request("POST", "/api/v1/analyze/document", {
      body: JSON.stringify(payload),
      idempotencyKey: options.idempotencyKey || generateIdempotencyKey(),
    });
  }

  async getAnalysisStatus(jobId) {
    return this.request("GET", `/api/v1/analyze/${jobId}`);
  }

  async getAnalysisResult(jobId) {
    return this.request("GET", `/api/v1/analyze/${jobId}/result`);
  }

  async waitForAnalysis(jobId, options = {}) {
    const { timeout = 300000, pollInterval = 2000 } = options;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      try {
        const result = await this.getAnalysisResult(jobId);
        return result;
      } catch (error) {
        if (error.message.includes("202")) {
          // Still processing
          await new Promise((resolve) => setTimeout(resolve, pollInterval));
        } else {
          throw error;
        }
      }
    }

    throw new Error(
      `Analysis did not complete within ${timeout / 1000} seconds`
    );
  }

  // ========================================================================
  // Case Search
  // ========================================================================

  async searchCases(options = {}) {
    const {
      keywords = [],
      jurisdiction = "US",
      limit = 10,
    } = options;

    return this.request("POST", "/api/v1/cases/search", {
      body: JSON.stringify({
        keywords,
        jurisdiction,
        limit,
      }),
      idempotencyKey: options.idempotencyKey || generateIdempotencyKey(),
    });
  }

  async getCaseTimeline(caseId) {
    return this.request("GET", `/api/v1/cases/${caseId}/timeline`);
  }

  // ========================================================================
  // Reports
  // ========================================================================

  async generateReport(options = {}) {
    const { caseId, reportType = "comprehensive", format = "pdf" } = options;

    if (!caseId) {
      throw new Error("caseId is required");
    }

    return this.request("POST", "/api/v1/reports/generate", {
      body: JSON.stringify({
        case_id: caseId,
        report_type: reportType,
        format,
      }),
      idempotencyKey: options.idempotencyKey || generateIdempotencyKey(),
    });
  }

  async getReportStatus(reportId) {
    return this.request("GET", `/api/v1/reports/${reportId}`);
  }

  async waitForReport(reportId, options = {}) {
    const { timeout = 600000 } = options;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const status = await this.getReportStatus(reportId);

      if (status.status === "completed") {
        return status.download_url;
      } else if (status.status === "failed") {
        throw new Error("Report generation failed");
      }

      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    throw new Error(`Report did not complete within ${timeout / 1000} seconds`);
  }

  // ========================================================================
  // Deadlines
  // ========================================================================

  async getUpcomingDeadlines(days = 30) {
    return this.request("GET", "/api/v1/deadlines/upcoming", {
      params: new URLSearchParams({ days }),
    });
  }

  async createDeadline(options = {}) {
    const {
      title,
      dueDate,
      description = "",
      priority = "medium",
    } = options;

    if (!title || !dueDate) {
      throw new Error("title and dueDate are required");
    }

    return this.request("POST", "/api/v1/deadlines", {
      body: JSON.stringify({
        title,
        due_date: dueDate,
        description,
        priority,
      }),
      idempotencyKey: options.idempotencyKey || generateIdempotencyKey(),
    });
  }

  // ========================================================================
  // Analytics
  // ========================================================================

  async getCostBreakdown(period = "monthly") {
    return this.request("GET", "/api/v1/analytics/costs", {
      params: new URLSearchParams({ period }),
    });
  }

  async getAnalyticsOverview() {
    return this.request("GET", "/api/v1/analytics/overview");
  }

  // ========================================================================
  // WebSocket Support
  // ========================================================================

  connectProgress(jobId, callback) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/progress/${jobId}`;

    const protocols = this.token ? ["access_token", this.token] : [];
    const ws = new WebSocket(wsUrl, protocols);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      callback(data);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return ws;
  }
}

// Export for Node.js and browser
if (typeof module !== "undefined" && module.exports) {
  module.exports = LegalassistClient;
}

// Example usage:
/*
const client = new LegalassistClient({ apiKey: 'your-api-key' });

// Analyze document
const result = await client.analyzeDocument({ text: 'Contract...' });
console.log('Job ID:', result.job_id);

// Wait for result
const analysis = await client.waitForAnalysis(result.job_id);
console.log('Summary:', analysis.summary);

// WebSocket progress tracking
const ws = client.connectProgress(result.job_id, (data) => {
  console.log(`Progress: ${data.progress}% - ${data.status}`);
});
*/
