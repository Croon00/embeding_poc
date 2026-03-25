from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ibm_boto3
import requests
from dotenv import load_dotenv
from ibm_botocore.client import Config as CosConfig
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from parse_documents import parse_document


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        if end < len(normalized):
            split = normalized.rfind(" ", start, end)
            if split > start + 100:
                end = split
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - chunk_overlap)
    return chunks


@dataclass
class Settings:
    cos_apikey: str
    cos_resource_instance_id: str
    cos_bucket: str
    cos_endpoint_url: str
    source_cos_prefix: str
    watsonx_apikey: str
    watsonx_url: str
    watsonx_project_id: str
    watsonx_version: str
    watsonx_embedding_model_id: str
    watsonx_embed_batch_size: int
    milvus_uri: str | None
    milvus_host: str | None
    milvus_port: int | None
    milvus_secure: bool
    milvus_username: str | None
    milvus_password: str | None
    milvus_token: str | None
    milvus_db_name: str | None
    milvus_collection_name: str
    milvus_consistency_level: str
    milvus_metric_type: str
    milvus_index_type: str
    milvus_ca_pem_path: str | None
    milvus_ca_cert: str | None
    milvus_server_pem_path: str | None
    milvus_server_name: str | None
    local_workdir: Path
    chunk_size: int
    chunk_overlap: int
    embed_only_attachments: bool
    delete_existing_ids: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        cos_endpoint_url = os.getenv("COS_ENDPOINT") or os.getenv("COS_ENDPOINTS_URL", "")
        source_prefix = os.getenv("SOURCE_COS_PREFIX", os.getenv("COS_PREFIX", "")).strip("/")
        return cls(
            cos_apikey=os.environ["COS_APIKEY"],
            cos_resource_instance_id=os.environ["COS_RESOURCE_INSTANCE_ID"],
            cos_bucket=os.environ["COS_BUCKET"],
            cos_endpoint_url=cos_endpoint_url,
            source_cos_prefix=source_prefix,
            watsonx_apikey=os.environ["WATSONX_APIKEY"],
            watsonx_url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/"),
            watsonx_project_id=os.environ["WATSONX_PROJECT_ID"],
            watsonx_version=os.getenv("WATSONX_VERSION", "2023-10-25"),
            watsonx_embedding_model_id=os.getenv(
                "WATSONX_EMBEDDING_MODEL_ID", "granite-embedding-278m-multilingual"
            ),
            watsonx_embed_batch_size=int(os.getenv("WATSONX_EMBED_BATCH_SIZE", "16")),
            milvus_uri=(os.getenv("MILVUS_URI") or "").strip() or None,
            milvus_host=(os.getenv("MILVUS_HOST") or "").strip() or None,
            milvus_port=int(os.getenv("MILVUS_PORT")) if os.getenv("MILVUS_PORT") else None,
            milvus_secure=os.getenv("MILVUS_SECURE", "false").lower() == "true",
            milvus_username=os.getenv("MILVUS_USERNAME") or None,
            milvus_password=os.getenv("MILVUS_PASSWORD") or None,
            milvus_token=os.getenv("MILVUS_TOKEN") or None,
            milvus_db_name=os.getenv("MILVUS_DB_NAME") or None,
            milvus_collection_name=os.getenv("MILVUS_COLLECTION_NAME", "policy_embed_index_job_chunks"),
            milvus_consistency_level=os.getenv("MILVUS_CONSISTENCY_LEVEL", "Bounded"),
            milvus_metric_type=os.getenv("MILVUS_METRIC_TYPE", "COSINE"),
            milvus_index_type=os.getenv("MILVUS_INDEX_TYPE", "AUTOINDEX"),
            milvus_ca_pem_path=(os.getenv("MILVUS_CA_PEM_PATH") or "").strip() or None,
            milvus_ca_cert=(os.getenv("MILVUS_CA_CERT") or "").strip() or None,
            milvus_server_pem_path=(os.getenv("MILVUS_SERVER_PEM_PATH") or "").strip() or None,
            milvus_server_name=(os.getenv("MILVUS_SERVER_NAME") or "").strip() or None,
            local_workdir=Path(os.getenv("LOCAL_WORKDIR", "/tmp/policy-embed-index-job")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            embed_only_attachments=os.getenv("EMBED_ONLY_ATTACHMENTS", "false").lower() == "true",
            delete_existing_ids=os.getenv("DELETE_EXISTING_IDS", "true").lower() == "true",
        )


def _materialize_cert_file(file_name: str, cert_value: str) -> str:
    target_path = Path(tempfile.gettempdir()) / file_name
    target_path.write_text(cert_value, encoding="utf-8")
    return str(target_path)


def build_milvus_connect_args(settings: Settings) -> dict[str, object]:
    if settings.milvus_uri:
        connect_args: dict[str, object] = {"alias": "default", "uri": settings.milvus_uri}
    elif settings.milvus_host and settings.milvus_port:
        connect_args = {
            "alias": "default",
            "host": settings.milvus_host,
            "port": settings.milvus_port,
        }
    else:
        raise RuntimeError("Set MILVUS_URI or both MILVUS_HOST and MILVUS_PORT")

    if settings.milvus_secure:
        connect_args["secure"] = True
    if settings.milvus_ca_pem_path:
        connect_args["ca_pem_path"] = settings.milvus_ca_pem_path
    elif settings.milvus_ca_cert:
        connect_args["ca_pem_path"] = _materialize_cert_file("milvus-ca.pem", settings.milvus_ca_cert)
    if settings.milvus_server_pem_path:
        connect_args["server_pem_path"] = settings.milvus_server_pem_path
    if settings.milvus_server_name:
        connect_args["server_name"] = settings.milvus_server_name

    use_username_password = bool(settings.milvus_username or settings.milvus_password)
    if settings.milvus_username:
        connect_args["user"] = settings.milvus_username
    if settings.milvus_password:
        connect_args["password"] = settings.milvus_password
    if settings.milvus_token and not use_username_password:
        connect_args["token"] = settings.milvus_token
    if settings.milvus_db_name:
        connect_args["db_name"] = settings.milvus_db_name
    return connect_args


class CosDownloader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=settings.cos_apikey,
            ibm_service_instance_id=settings.cos_resource_instance_id,
            config=CosConfig(signature_version="oauth"),
            endpoint_url=settings.cos_endpoint_url,
        )

    def sync_prefix(self, local_root: Path) -> list[Path]:
        prefix = self.settings.source_cos_prefix
        if prefix:
            prefix += "/"

        downloaded: list[Path] = []
        continuation_token: str | None = None

        while True:
            kwargs = {
                "Bucket": self.settings.cos_bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = self.client.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                relative_key = key[len(prefix) :] if prefix and key.startswith(prefix) else key
                target_path = local_root / relative_key
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.settings.cos_bucket, key, str(target_path))
                downloaded.append(target_path)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        return downloaded


class WatsonxEmbeddingsClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

    def _clear_token(self) -> None:
        self._access_token = None
        self._access_token_expires_at = 0.0

    def _token(self, force_refresh: bool = False) -> str:
        now = time.time()
        if (
            not force_refresh
            and self._access_token
            and now < self._access_token_expires_at - 60
        ):
            return self._access_token

        response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.settings.watsonx_apikey,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 3600))
        self._access_token_expires_at = now + expires_in
        return self._access_token

    def _post_embeddings(self, texts: list[str], force_refresh: bool = False) -> requests.Response:
        return requests.post(
            (
                f"{self.settings.watsonx_url}/ml/v1/text/embeddings"
                f"?version={self.settings.watsonx_version}"
            ),
            headers={
                "Authorization": f"Bearer {self._token(force_refresh=force_refresh)}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "inputs": texts,
                "model_id": self.settings.watsonx_embedding_model_id,
                "project_id": self.settings.watsonx_project_id,
            },
            timeout=120,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._post_embeddings(texts)
        if response.status_code == 401 and "authentication_token_expired" in response.text:
            self._clear_token()
            response = self._post_embeddings(texts, force_refresh=True)
        if not response.ok:
            sample_lengths = [len(text) for text in texts[:5]]
            raise RuntimeError(
                "watsonx embeddings request failed: "
                f"status={response.status_code}, "
                f"batch_size={len(texts)}, "
                f"sample_text_lengths={sample_lengths}, "
                f"body={response.text}"
            )
        response.raise_for_status()
        payload = response.json()
        return [row["embedding"] for row in payload["results"]]


class MilvusWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        connections.connect(**build_milvus_connect_args(settings))
        self.collection: Collection | None = None

    def ensure_collection(self, vector_dim: int) -> Collection:
        name = self.settings.milvus_collection_name
        if utility.has_collection(name):
            self.collection = Collection(name=name, using="default")
            return self.collection

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="article_url", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Policy chunks embedded by policy-embed-index-job",
        )
        collection = Collection(
            name=name,
            schema=schema,
            using="default",
            consistency_level=self.settings.milvus_consistency_level,
        )
        collection.create_index(
            field_name="vector",
            index_params={"index_type": self.settings.milvus_index_type, "metric_type": self.settings.milvus_metric_type, "params": {}},
        )
        self.collection = collection
        return collection

    def upsert(self, rows: list[dict], vectors: list[list[float]]) -> None:
        if not rows:
            return
        if self.collection is None:
            self.ensure_collection(len(vectors[0]))
        assert self.collection is not None

        if self.settings.delete_existing_ids:
            for start in range(0, len(rows), 200):
                ids = [row["chunk_id"] for row in rows[start : start + 200]]
                quoted = ", ".join(json.dumps(value) for value in ids)
                self.collection.delete(expr=f"id in [{quoted}]")

        entities = [
            [row["chunk_id"] for row in rows],
            [row["document_id"] for row in rows],
            [row["source"] for row in rows],
            [row["source_type"] for row in rows],
            [row["title"][:2048] for row in rows],
            [row["article_url"][:2048] for row in rows],
            [row["source_path"][:2048] for row in rows],
            [row["text"][:65535] for row in rows],
            [json.dumps(row["metadata"], ensure_ascii=False)[:65535] for row in rows],
            vectors,
        ]
        self.collection.insert(entities)
        self.collection.flush()


def load_articles(output_dir: Path) -> list[dict]:
    articles: list[dict] = []
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                articles.append(json.loads(line))
    return articles


def build_document_records(settings: Settings, output_dir: Path, downloads_dir: Path) -> list[dict]:
    documents: list[dict] = []
    articles = load_articles(output_dir)

    for article in articles:
        article_url = article.get("url", "")
        article_source = article.get("source", "unknown")
        article_doc_id = f"article-{stable_id(article_source, article_url)}"

        if not settings.embed_only_attachments:
            article_text = clean_text(article.get("body"))
            if article_text:
                documents.append(
                    {
                        "document_id": article_doc_id,
                        "parent_document_id": None,
                        "source": article_source,
                        "source_type": "article_body",
                        "title": article.get("title") or article_url,
                        "text": article_text,
                        "source_path": f"output/{article_source}.jsonl",
                        "article_url": article_url,
                        "metadata": {
                            "date": article.get("date"),
                            "board_name": article.get("board_name"),
                            "section_name": article.get("section_name"),
                            "search_keyword": article.get("search_keyword"),
                        },
                    }
                )

        for index, file_info in enumerate(article.get("files") or [], start=1):
            relative_path = file_info.get("path")
            if not relative_path:
                continue
            local_path = downloads_dir / relative_path
            if not local_path.exists():
                continue

            parsed = parse_document(local_path)
            text = clean_text(parsed.get("full_text"))
            if not text:
                continue

            attachment_doc_id = f"attachment-{stable_id(article_url, relative_path)}"
            documents.append(
                {
                    "document_id": attachment_doc_id,
                    "parent_document_id": article_doc_id,
                    "source": article_source,
                    "source_type": "attachment",
                    "title": local_path.name,
                    "text": text,
                    "source_path": str(local_path),
                    "article_url": article_url,
                    "metadata": {
                        "attachment_index": index,
                        "parser": parsed.get("parser"),
                        "size_bytes": parsed.get("size_bytes"),
                        "attachment_path": relative_path,
                    },
                }
            )

            for child_index, child in enumerate(parsed.get("parsed_children", []), start=1):
                child_path = child.get("path")
                if not child_path:
                    continue
                child_parsed = parse_document(child_path)
                child_text = clean_text(child_parsed.get("full_text"))
                if not child_text:
                    continue
                documents.append(
                    {
                        "document_id": f"attachment-child-{stable_id(article_url, child_path)}",
                        "parent_document_id": attachment_doc_id,
                        "source": article_source,
                        "source_type": "attachment_child",
                        "title": child.get("name") or Path(child_path).name,
                        "text": child_text,
                        "source_path": child_path,
                        "article_url": article_url,
                        "metadata": {
                            "attachment_index": index,
                            "child_index": child_index,
                            "attachment_path": relative_path,
                            "parser": child_parsed.get("parser"),
                            "size_bytes": child_parsed.get("size_bytes"),
                        },
                    }
                )

    return documents


def build_chunk_records(documents: Iterable[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks: list[dict] = []

    for document in documents:
        text = clean_text(document.get("text"))
        if not text:
            continue

        for index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "chunk_id": f"{document['document_id']}-chunk-{index:04d}",
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "source_type": document["source_type"],
                    "title": document["title"],
                    "text": chunk,
                    "source_path": document["source_path"],
                    "article_url": document["article_url"],
                    "metadata": {
                        **document["metadata"],
                        "parent_document_id": document["parent_document_id"],
                        "chunk_index": index,
                        "char_count": len(chunk),
                        "token_estimate": math.ceil(len(chunk) / 4),
                    },
                }
            )

    return chunks


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    settings = Settings.from_env()
    settings.local_workdir.mkdir(parents=True, exist_ok=True)
    input_root = settings.local_workdir / "input"
    output_dir = input_root / "output"
    downloads_dir = input_root / "downloads"

    print(
        f"[policy-embed-index-job] COS sync start: "
        f"bucket={settings.cos_bucket}, prefix={settings.source_cos_prefix or '/'}"
    )
    downloader = CosDownloader(settings)
    downloaded_files = downloader.sync_prefix(input_root)
    print(f"[policy-embed-index-job] COS sync done: {len(downloaded_files)} files")

    if not output_dir.exists():
        raise RuntimeError(f"Expected output directory not found after COS sync: {output_dir}")

    documents = build_document_records(settings, output_dir, downloads_dir)
    chunks = build_chunk_records(documents, settings.chunk_size, settings.chunk_overlap)
    print(f"[policy-embed-index-job] Built {len(documents)} documents and {len(chunks)} chunks")

    if not chunks:
        raise RuntimeError("No chunks were produced from COS inputs.")

    embedding_client = WatsonxEmbeddingsClient(settings)
    milvus_writer = MilvusWriter(settings)

    total = len(chunks)
    for start in range(0, total, settings.watsonx_embed_batch_size):
        batch = chunks[start : start + settings.watsonx_embed_batch_size]
        vectors = embedding_client.embed_texts([row["text"] for row in batch])
        milvus_writer.upsert(batch, vectors)
        print(
            "[policy-embed-index-job] Embedded and upserted "
            f"{min(start + len(batch), total)}/{total} chunks"
        )

    report = {
        "downloaded_file_count": len(downloaded_files),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "cos_bucket": settings.cos_bucket,
        "source_prefix": settings.source_cos_prefix,
        "watsonx_model_id": settings.watsonx_embedding_model_id,
        "milvus_collection_name": settings.milvus_collection_name,
    }
    write_json(settings.local_workdir / "policy-embed-index-job-report.json", report)
    print("[policy-embed-index-job] Complete")


if __name__ == "__main__":
    main()
