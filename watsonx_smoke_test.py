from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    load_dotenv()

    apikey = required("WATSONX_APIKEY")
    project_id = required("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
    version = os.getenv("WATSONX_VERSION", "2023-10-25")
    model_id = os.getenv("WATSONX_EMBEDDING_MODEL_ID", "granite-embedding-278m-multilingual")

    print(f"[watsonx-test] url={url}")
    print(f"[watsonx-test] project_id={project_id}")
    print(f"[watsonx-test] model_id={model_id}")
    print(f"[watsonx-test] version={version}")

    token_response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": apikey,
        },
        timeout=30,
    )
    print(f"[watsonx-test] iam status={token_response.status_code}")
    if not token_response.ok:
        print(token_response.text)
        token_response.raise_for_status()

    access_token = token_response.json()["access_token"]
    embed_response = requests.post(
        f"{url}/ml/v1/text/embeddings?version={version}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "inputs": ["한국 정책 문서 임베딩 테스트"],
            "model_id": model_id,
            "project_id": project_id,
        },
        timeout=120,
    )
    print(f"[watsonx-test] embeddings status={embed_response.status_code}")
    print(embed_response.text)
    embed_response.raise_for_status()
    print("[watsonx-test] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[watsonx-test] FAILED -> {exc}", file=sys.stderr)
        raise
