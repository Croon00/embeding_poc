from __future__ import annotations

import os
import sys
import tempfile

from dotenv import load_dotenv
from pymilvus import connections, utility


def main() -> int:
    load_dotenv()

    uri = os.getenv("MILVUS_URI") or None
    host = os.getenv("MILVUS_HOST") or None
    port = os.getenv("MILVUS_PORT") or None
    secure = os.getenv("MILVUS_SECURE", "false").lower() == "true"
    username = os.getenv("MILVUS_USERNAME") or None
    password = os.getenv("MILVUS_PASSWORD") or None
    token = os.getenv("MILVUS_TOKEN") or None
    db_name = os.getenv("MILVUS_DB_NAME") or None
    ca_pem_path = os.getenv("MILVUS_CA_PEM_PATH") or None
    ca_cert = os.getenv("MILVUS_CA_CERT") or None
    server_pem_path = os.getenv("MILVUS_SERVER_PEM_PATH") or None
    server_name = os.getenv("MILVUS_SERVER_NAME") or None

    if uri:
        connect_args = {"alias": "default", "uri": uri}
    elif host and port:
        connect_args = {"alias": "default", "host": host, "port": int(port)}
    else:
        raise RuntimeError("Set MILVUS_URI or both MILVUS_HOST and MILVUS_PORT")

    if secure:
        connect_args["secure"] = True
    if ca_pem_path:
        connect_args["ca_pem_path"] = ca_pem_path
    elif ca_cert:
        target_path = os.path.join(tempfile.gettempdir(), "milvus-ca.pem")
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write(ca_cert)
        connect_args["ca_pem_path"] = target_path
    if server_pem_path:
        connect_args["server_pem_path"] = server_pem_path
    if server_name:
        connect_args["server_name"] = server_name

    use_username_password = bool(username or password)
    if username:
        connect_args["user"] = username
    if password:
        connect_args["password"] = password
    if token and not use_username_password:
        connect_args["token"] = token
    if db_name:
        connect_args["db_name"] = db_name

    if uri:
        print(f"[milvus-test] uri={uri}")
    else:
        print(f"[milvus-test] host={host}")
        print(f"[milvus-test] port={port}")
        print(f"[milvus-test] secure={secure}")
    if username:
        print(f"[milvus-test] username={username}")
    if db_name:
        print(f"[milvus-test] db_name={db_name}")

    try:
        connections.connect(**connect_args)
        print("[milvus-test] connect: OK")

        collections = utility.list_collections(using="default")
        print(f"[milvus-test] list_collections: OK ({len(collections)} collections)")
        for name in collections[:20]:
            print(f" - {name}")
        return 0
    except Exception as exc:
        print(f"[milvus-test] FAILED -> {exc}", file=sys.stderr)
        raise
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
