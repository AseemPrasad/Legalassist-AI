# WebSocket Integration Guide

## Overview

LegalAssist AI provides real-time WebSocket endpoints for case timeline updates, job progress tracking, and collaborative features. This guide explains how to connect, authenticate, and handle events.

## Connection

### Endpoint

Production: wss://api.legalassist-ai.io/ws
Local development: ws://localhost:8000/ws

### Authentication

WebSocket connections authenticate via JWT token passed as a query parameter:

`javascript
const ws = new WebSocket(wss://api.legalassist-ai.io/ws?token=);
`

The token must be obtained from the authentication endpoint (POST /api/v1/auth/token).

## Event Types

### Case Timeline Events

| Event | Direction | Description |
|-------|-----------|-------------|
| case.updated | Server → Client | Case metadata changed |
| 	imeline.event_added | Server → Client | New timeline entry created |
| deadline.approaching | Server → Client | Deadline is approaching (configurable threshold) |

### Job Progress Events

| Event | Direction | Description |
|-------|-----------|-------------|
| job.started | Server → Client | Background job began processing |
| job.progress | Server → Client | Job progress update with percentage |
| job.completed | Server → Client | Job finished successfully |
| job.failed | Server → Client | Job failed with error details |

### Payload Format

All events follow this structure:

`json
{
  "event": "job.progress",
  "data": {
    "job_id": "abc-123",
    "progress": 75,
    "message": "Processing document page 3/4"
  },
  "timestamp": "2026-06-15T10:30:00Z"
}
`

## Reconnection Strategy

The client should implement exponential backoff reconnection:

1. Initial delay: 1 second
2. Backoff multiplier: 2x
3. Maximum delay: 30 seconds
4. Maximum retries: 10

### JavaScript Example

`javascript
class LegalAssistWebSocket {
  constructor(token, options = {}) {
    this.token = token;
    this.reconnectAttempts = 0;
    this.maxRetries = options.maxRetries || 10;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 30000;
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(wss://api.legalassist-ai.io/ws?token=);
    this.ws.onopen = () => { this.reconnectAttempts = 0; };
    this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));
    this.ws.onclose = () => this.reconnect();
    this.ws.onerror = () => this.ws.close();
  }

  reconnect() {
    if (this.reconnectAttempts >= this.maxRetries) return;
    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.reconnectAttempts),
      this.maxDelay
    );
    setTimeout(() => { this.reconnectAttempts++; this.connect(); }, delay);
  }

  handleMessage(msg) {
    console.log(Event: , msg.data);
  }
}
`

### Python Example

`python
import asyncio
import websockets
import json

async def connect_legalassist_ws(token):
    uri = f"wss://api.legalassist-ai.io/ws?token={token}"
    async for ws in websockets.connect(uri):
        try:
            async for message in ws:
                event = json.loads(message)
                print(f"Event: {event['event']}", event['data'])
        except websockets.ConnectionClosed:
            continue
`

## Error Handling

| Close Code | Meaning | Action |
|------------|---------|--------|
| 4001 | Invalid token | Refresh token and reconnect |
| 4002 | Token expired | Obtain new token |
| 4003 | Rate limited | Wait and retry with backoff |
| 4004 | Session limit reached | Close other connections |

## Architecture

The WebSocket infrastructure consists of:

- pi/websockets/router.py — Route messages to handlers
- pi/websockets/connection_manager.py — Track and manage connections
- pi/websockets/reconnection.py — Handle reconnection logic
