<<<<<<< HEAD
# policy-embed-index-job

`policy-embed-index-job` is a separate Code Engine ingestion job that reads files uploaded by `crawler_POC` from IBM Cloud Object Storage, extracts text from article attachments, generates multilingual embeddings with watsonx.ai Granite, and writes chunk vectors to Milvus.

## Flow

1. Download `output/*.jsonl` and `downloads/**` from COS.
2. Build document records from article bodies and downloaded attachments.
3. Extract text from `pdf`, `hwp`, `hwpx`, and nested zip children.
4. Chunk text into retrieval-sized passages.
5. Call watsonx.ai embeddings with `granite-embedding-278m-multilingual`.
6. Insert chunk vectors and metadata into Milvus.

## Required environment variables

### COS

- `COS_APIKEY`
- `COS_RESOURCE_INSTANCE_ID`
- `COS_BUCKET`
- `COS_ENDPOINTS_URL`
- `SOURCE_COS_PREFIX`

### watsonx.ai

- `WATSONX_APIKEY`
- `WATSONX_URL`
- `WATSONX_PROJECT_ID`
- `WATSONX_EMBEDDING_MODEL_ID`

### Milvus

- `MILVUS_URI`
- `MILVUS_USERNAME` optional
- `MILVUS_PASSWORD` optional
- `MILVUS_TOKEN` optional
- `MILVUS_DB_NAME` optional
- `MILVUS_COLLECTION_NAME`

Default recommendation for watsonx.data Milvus:

```env
MILVUS_URI=https://<milvus-https-host>:31435
MILVUS_USERNAME=ibmlhapikey_<your_account_email>
MILVUS_PASSWORD=<ibm_cloud_apikey>
MILVUS_TOKEN=
```

If `MILVUS_USERNAME` or `MILVUS_PASSWORD` is set, `MILVUS_TOKEN` is ignored.

Alternative gRPC/TLS configuration:

```env
MILVUS_URI=
MILVUS_HOST=<milvus-grpc-host>
MILVUS_PORT=30246
MILVUS_SECURE=true
MILVUS_CA_PEM_PATH=/path/to/ca.pem
MILVUS_USERNAME=<milvus-username>
MILVUS_PASSWORD=<milvus-password>
```

## Local run

```powershell
cd C:\motie-adk-project\jobB
copy .env.example .env
python -m pip install -r requirements.txt
python run_job.py
```

## Milvus connection test

```powershell
cd C:\motie-adk-project\jobB
copy .env.example .env
python milvus_smoke_test.py
```

## Docker build

```powershell
cd C:\motie-adk-project\jobB
docker build -t us.icr.io/<namespace>/policy-embed-index-job:latest .
```

## Code Engine

Use this directory as a separate image and job from `crawler_POC`.

```powershell
ibmcloud ce job create --name policy-embed-index-job --image us.icr.io/<namespace>/policy-embed-index-job:latest --registry-secret icr-pull-secret
```

Then attach a secret or env set that contains the COS, watsonx.ai, and Milvus variables from `.env.example`.
=======
# embeding_poc
embeding 과정 실습
>>>>>>> cdbad8a787d6ceb3ef1f57d5b670f8f0cfb49d7c
