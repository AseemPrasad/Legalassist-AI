"""
Grafana dashboard provisioning script
Automatically creates production-grade dashboards
"""

import json
import requests
import os
from typing import Dict, Any


class GrafanaDashboardProvisioner:
    def __init__(self, grafana_url: str, api_token: str):
        self.url = grafana_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
    
    def create_dashboard(self, dashboard_name: str, dashboard_config: Dict[str, Any]) -> int:
        """Create or update dashboard, returns dashboard ID"""
        response = requests.post(
            f"{self.url}/api/dashboards/db",
            headers=self.headers,
            json={"dashboard": dashboard_config, "overwrite": True},
        )
        response.raise_for_status()
        return response.json()["id"]
    
    def create_datasource(self, ds_name: str, ds_type: str, ds_url: str) -> str:
        """Create datasource, returns datasource ID"""
        payload = {
            "name": ds_name,
            "type": ds_type,
            "url": ds_url,
            "access": "proxy",
            "isDefault": ds_type == "prometheus",
        }
        
        response = requests.post(
            f"{self.url}/api/datasources",
            headers=self.headers,
            json=payload,
        )
        
        if response.status_code == 409:
            # Already exists
            return response.json()["id"]
        
        response.raise_for_status()
        return response.json()["id"]
    
    def provision_all(self):
        """Create all production dashboards and datasources"""
        
        # Create datasources
        prometheus_id = self.create_datasource("Prometheus", "prometheus", "http://prometheus:9090")
        elasticsearch_id = self.create_datasource("Elasticsearch", "elasticsearch", "http://elasticsearch:9200")
        
        # Create dashboards (defined in next section)
        self.create_system_overview_dashboard()
        self.create_application_metrics_dashboard()
        self.create_llm_usage_dashboard()
        self.create_database_health_dashboard()
        self.create_business_metrics_dashboard()
        
        print("✅ Grafana dashboards provisioned successfully!")


def create_system_overview_dashboard() -> Dict:
    """System resources dashboard"""
    return {
        "title": "System Overview",
        "description": "CPU, memory, network, disk utilization",
        "panels": [
            {
                "title": "CPU Usage (%)",
                "targets": [{"expr": "rate(container_cpu_usage_seconds_total[5m]) * 100"}],
            },
            {
                "title": "Memory Usage (GB)",
                "targets": [{"expr": "container_memory_usage_bytes / 1e9"}],
            },
            {
                "title": "Network In (KB/s)",
                "targets": [{"expr": "rate(container_network_receive_bytes[5m]) / 1024"}],
            },
            {
                "title": "Disk Used (%)",
                "targets": [{"expr": "(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100"}],
            },
        ],
    }


def create_application_metrics_dashboard() -> Dict:
    """Application performance metrics"""
    return {
        "title": "Application Metrics",
        "description": "Request rate, latency, errors",
        "panels": [
            {
                "title": "Request Rate (req/s)",
                "targets": [{"expr": "rate(http_requests_total[1m])"}],
            },
            {
                "title": "P95 Latency (ms)",
                "targets": [{"expr": "histogram_quantile(0.95, http_request_duration_seconds) * 1000"}],
            },
            {
                "title": "Error Rate (%)",
                "targets": [{"expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100"}],
            },
            {
                "title": "Active Connections",
                "targets": [{"expr": "http_requests_queued"}],
            },
        ],
    }


def create_llm_usage_dashboard() -> Dict:
    """LLM API usage and costs"""
    return {
        "title": "LLM Usage & Costs",
        "description": "Token usage, API costs, error rates",
        "panels": [
            {
                "title": "Daily Token Usage",
                "targets": [{"expr": "increase(llm_tokens_used_total[1d])"}],
            },
            {
                "title": "Cumulative Costs ($)",
                "targets": [{"expr": "llm_api_costs_total"}],
            },
            {
                "title": "LLM Error Rate (%)",
                "targets": [{"expr": "rate(llm_api_calls_total{status=\"error\"}[5m]) / rate(llm_api_calls_total[5m]) * 100"}],
            },
            {
                "title": "API Call Duration (ms)",
                "targets": [{"expr": "histogram_quantile(0.95, llm_api_call_duration_seconds) * 1000"}],
            },
        ],
    }


def create_database_health_dashboard() -> Dict:
    """Database performance and health"""
    return {
        "title": "Database Health",
        "description": "Connections, query latency, slow queries",
        "panels": [
            {
                "title": "Active Connections",
                "targets": [{"expr": "db_active_connections"}],
            },
            {
                "title": "Query Latency P95 (ms)",
                "targets": [{"expr": "histogram_quantile(0.95, db_query_duration_seconds) * 1000"}],
            },
            {
                "title": "Connection Pool Usage (%)",
                "targets": [{"expr": "db_active_connections / db_connection_pool_size * 100"}],
            },
            {
                "title": "Slow Queries (>1s)",
                "targets": [{"expr": "rate(db_query_duration_seconds_bucket{le=\"1.0\"}[5m])"}],
            },
        ],
    }


def create_business_metrics_dashboard() -> Dict:
    """Business KPIs"""
    return {
        "title": "Business Metrics",
        "description": "Active cases, deadlines, user sessions",
        "panels": [
            {
                "title": "Active Cases",
                "targets": [{"expr": "active_cases"}],
            },
            {
                "title": "Pending Deadlines",
                "targets": [{"expr": "pending_deadlines"}],
            },
            {
                "title": "Active User Sessions",
                "targets": [{"expr": "user_sessions_active"}],
            },
            {
                "title": "Document Processing Success Rate (%)",
                "targets": [{"expr": "rate(document_processing_total{status=\"success\"}[5m]) / rate(document_processing_total[5m]) * 100"}],
            },
        ],
    }


if __name__ == "__main__":
    grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
    api_token = os.getenv("GRAFANA_API_TOKEN")
    if not api_token:
        raise RuntimeError("GRAFANA_API_TOKEN environment variable is required")
    
    provisioner = GrafanaDashboardProvisioner(grafana_url, api_token)
    provisioner.provision_all()
