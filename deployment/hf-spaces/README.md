---
title: Centurion Core API
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Centurion Core — Backend API

Algorithmic trading platform backend powered by FastAPI.

## Environment Variables (set in HF Spaces Settings → Secrets)

| Variable | Required | Description |
|----------|----------|-------------|
| `CENTURION_DATABASE_URL` | Yes | Neon PostgreSQL pooler URL |
| `ANTHROPIC_API_KEY` | Yes | Claude API key for RAG |
| `ZERODHA_API_KEY` | Yes | Kite Connect API key |
| `ZERODHA_API_SECRET` | Yes | Kite Connect secret |
| `MINIO_ENDPOINT` | Yes | Cloudflare R2 endpoint |
| `MINIO_ACCESS_KEY` | Yes | R2 access key |
| `MINIO_SECRET_KEY` | Yes | R2 secret key |
| `UPSTASH_REDIS_URL` | Optional | Upstash Redis URL |
| `CENTURION_CREDENTIALS_YAML` | Yes | Login users and bcrypt hashes (the `auth/credentials.yaml` format); set by the deploy workflow from the `CREDENTIALS_YAML` GitHub secret, never committed here |
| `CENTURION_API_SECRET_KEY` | Yes | 32+ random bytes (hex) signing sign-in tokens; unset, tokens end at every restart |
| `CENTURION_KITE_USER_ID` | Yes | Your Zerodha user ID: the Kite login callback accepts no other |
| `CENTURION_ALLOWED_ORIGINS` | Yes | Comma-separated frontend URLs for CORS |
