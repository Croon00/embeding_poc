# embeding_POC

`embeding_POC`는 `crawler_POC`가 IBM Cloud Object Storage(COS)에 업로드한 게시글 본문과 첨부파일을 읽어, 텍스트를 추출하고 청크로 분할한 뒤 watsonx.ai 임베딩을 생성하여 Milvus에 적재하는 임베딩 파이프라인입니다.

현재 파이프라인은 다음 순서로 동작합니다.

1. COS에서 `output/*.jsonl`과 `downloads/**` 파일을 내려받습니다.
2. 게시글 본문과 첨부파일에서 문서 레코드를 생성합니다.
3. `pdf`, `hwp`, `hwpx`, `zip` 내부 파일과 일반 텍스트 파일에서 텍스트를 추출합니다.
4. 검색에 적합한 크기로 텍스트를 청크로 분할합니다.
5. watsonx.ai의 `granite-embedding-278m-multilingual` 모델로 임베딩을 생성합니다.
6. 청크 벡터와 메타데이터를 Milvus 컬렉션에 저장합니다.

## 주요 파일

- `run_job.py`: 전체 수집, 문서 생성, 청킹, 임베딩, Milvus 적재를 수행하는 메인 실행 파일
- `parse_documents.py`: 첨부파일에서 텍스트를 추출하는 파서 모듈
- `.env.example`: 실행에 필요한 환경 변수 예시
- `milvus_smoke_test.py`: Milvus 연결 확인용 스모크 테스트
- `watsonx_smoke_test.py`: watsonx.ai 임베딩 API 호출 확인용 스모크 테스트

## 지원하는 문서 형식

- 본문 텍스트
- `txt`, `csv`, `tsv`, `json`, `xml`, `html`, `htm`, `md`, `log`
- `pdf`
- `hwp`
- `hwpx`
- `zip` 내부의 지원 형식 파일

참고:
- `pdf`, `hwp`는 외부 전용 파서 없이 동작하므로 문서 구조에 따라 텍스트 추출이 제한될 수 있습니다.
- `zip` 파일은 내부 파일을 임시로 풀어서 재귀적으로 파싱합니다.

## 환경 변수

### 1. COS

- `COS_APIKEY`
- `COS_RESOURCE_INSTANCE_ID`
- `COS_BUCKET`
- `COS_ENDPOINTS_URL`
- `SOURCE_COS_PREFIX`

`SOURCE_COS_PREFIX`는 `crawler_POC`가 업로드한 결과가 저장된 prefix입니다. 예시는 `.env.example`의 `crawler_poc`입니다.

### 2. watsonx.ai

- `WATSONX_APIKEY`
- `WATSONX_URL`
- `WATSONX_PROJECT_ID`
- `WATSONX_VERSION`
- `WATSONX_EMBEDDING_MODEL_ID`
- `WATSONX_EMBED_BATCH_SIZE`

기본 임베딩 모델은 `granite-embedding-278m-multilingual`입니다.

### 3. Milvus

- `MILVUS_URI`
- `MILVUS_HOST`
- `MILVUS_PORT`
- `MILVUS_SECURE`
- `MILVUS_USERNAME`
- `MILVUS_PASSWORD`
- `MILVUS_TOKEN`
- `MILVUS_DB_NAME`
- `MILVUS_CA_PEM_PATH`
- `MILVUS_CA_CERT`
- `MILVUS_SERVER_PEM_PATH`
- `MILVUS_SERVER_NAME`
- `MILVUS_COLLECTION_NAME`
- `MILVUS_CONSISTENCY_LEVEL`
- `MILVUS_METRIC_TYPE`
- `MILVUS_INDEX_TYPE`

`MILVUS_URI`를 쓰거나, 또는 `MILVUS_HOST`와 `MILVUS_PORT`를 함께 설정해야 합니다.

권장 예시:

```env
MILVUS_URI=https://<milvus-https-host>:31435
MILVUS_USERNAME=ibmlhapikey_<your_account_email>
MILVUS_PASSWORD=<ibm_cloud_apikey>
MILVUS_TOKEN=
```

`MILVUS_USERNAME` 또는 `MILVUS_PASSWORD`가 설정되어 있으면 `MILVUS_TOKEN`은 사용되지 않습니다.

### 4. 작업 옵션

- `LOCAL_WORKDIR`: COS에서 내려받은 파일과 실행 결과를 저장할 로컬 작업 디렉터리
- `CHUNK_SIZE`: 청크 최대 문자 수
- `CHUNK_OVERLAP`: 청크 간 겹침 문자 수
- `EMBED_ONLY_ATTACHMENTS`: `true`면 게시글 본문은 제외하고 첨부파일만 임베딩
- `DELETE_EXISTING_IDS`: `true`면 동일 chunk id가 있을 때 먼저 삭제 후 다시 적재

## 로컬 실행 방법

```powershell
cd C:\jobA\embeding_POC
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python run_job.py
```

실행이 끝나면 `LOCAL_WORKDIR` 아래에 다운로드 파일과 `policy-embed-index-job-report.json` 리포트가 생성됩니다.

## 연결 테스트

Milvus 연결 확인:

```powershell
cd C:\jobA\embeding_POC
python milvus_smoke_test.py
```

watsonx.ai 임베딩 API 확인:

```powershell
cd C:\jobA\embeding_POC
python watsonx_smoke_test.py
```

## Docker 이미지 빌드

```powershell
cd C:\jobA\embeding_POC
docker build -t us.icr.io/<namespace>/policy-embed-index-job:latest .
```

## Code Engine Job 배포

이 디렉터리는 `crawler_POC`와 별도의 이미지 및 Job으로 배포하는 것을 기준으로 작성되어 있습니다.

```powershell
ibmcloud ce job create --name policy-embed-index-job --image us.icr.io/<namespace>/policy-embed-index-job:latest --registry-secret icr-pull-secret
```

이후 `.env.example`에 정의된 COS, watsonx.ai, Milvus 환경 변수를 Secret 또는 env로 연결하면 됩니다.

## 입력 데이터 형태

`crawler_POC`가 생성한 다음 구조를 기준으로 처리합니다.

- `output/*.jsonl`: 게시글 메타데이터와 본문
- `downloads/**`: 게시글 첨부파일

게시글 본문은 `article_body`, 첨부파일은 `attachment`, zip 내부에서 추출된 하위 파일은 `attachment_child` 유형으로 저장됩니다.

## 적재 결과

Milvus에는 다음 정보가 함께 저장됩니다.

- chunk id
- document id
- source
- source type
- title
- article url
- source path
- chunk text
- metadata json
- embedding vector

컬렉션이 없으면 실행 시 자동으로 생성되며, 벡터 인덱스도 함께 생성됩니다.
