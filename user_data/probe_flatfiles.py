import boto3
from botocore.config import Config

ACCESS_KEY = "2c7ed54b-3aeb-42ae-a7f4-c11fabda357c"
SECRET_KEY = "coinapi"

ENDPOINT_URL = "https://s3.flatfiles.coinapi.io"
BUCKET_NAME = "coinapi"
PREFIX = ""

KEYWORDS = ["BINANCE", "BTC", "USDT", "limitbook"]
MAX_PRINT = 200


def match_any(key: str) -> bool:
    k = key.lower()
    return any(word.lower() in k for word in KEYWORDS)


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    scanned = 0

    print("开始探测远端文件名...\n")

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            scanned += 1
            key = obj["Key"]
            if match_any(key):
                print(key)
                count += 1
                if count >= MAX_PRINT:
                    print(f"\n已输出 {MAX_PRINT} 条，停止。")
                    print(f"总共扫描对象数约: {scanned}")
                    return

    print(f"\n探测完成。")
    print(f"扫描对象数约: {scanned}")
    print(f"匹配到: {count} 条")
    

if __name__ == "__main__":
    main()
