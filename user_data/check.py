import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

ACCESS_KEY = "2c7ed54b-3aeb-42ae-a7f4-c11fabda357c"
SECRET_KEY = "coinapi"
ENDPOINT_URL = "https://s3.flatfiles.coinapi.io"
BUCKET_NAME = "coinapi"

TEST_PREFIXES = [
    "",
    "T-LIMITBOOK_FULL/",
    "T-TRADE/",
    "T-OHLCV/",
]

def is_real_file(obj: dict) -> bool:
    key = obj.get("Key", "")
    size = obj.get("Size", 0)

    # 排除目录标记，例如 T-LIMITBOOK_FULL/
    if key.endswith("/"):
        return False

    # 一般真实文件会有大小
    if size is None or size <= 0:
        return False

    return True

def main():
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )

    any_request_success = False
    real_files_found = []

    for prefix in TEST_PREFIXES:
        print("\n" + "=" * 70)
        print(f"测试前缀: {prefix or '(root)'}")
        print("=" * 70)

        try:
            resp = s3.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=prefix,
                MaxKeys=20
            )
            any_request_success = True

            contents = resp.get("Contents", [])
            print(f"原始返回对象数: {len(contents)}")

            if contents:
                print("原始返回 key:")
                for obj in contents:
                    print(f" - {obj.get('Key')}  size={obj.get('Size')}")
            else:
                print("没有返回任何对象")

            real_files = [obj for obj in contents if is_real_file(obj)]

            print(f"真实文件数: {len(real_files)}")
            for obj in real_files:
                print(f" [REAL] {obj['Key']} size={obj['Size']}")

            real_files_found.extend(real_files)

        except NoCredentialsError:
            print("错误：没有可用凭证")
            return
        except EndpointConnectionError as e:
            print("错误：连不上 endpoint")
            print(str(e))
            return
        except ClientError as e:
            err = e.response.get("Error", {})
            code = err.get("Code", "Unknown")
            msg = err.get("Message", str(e))
            print(f"ClientError: {code}: {msg}")

            if code in ["InvalidAccessKeyId", "SignatureDoesNotMatch", "AuthorizationHeaderMalformed"]:
                print("判断：AK/SK 不正确")
                return

            if code in ["AccessDenied", "AllAccessDisabled"]:
                print("判断：没有 Flat Files 权限")
                return

            return
        except Exception as e:
            print("未知异常：", str(e))
            return

    print("\n" + "=" * 70)
    print("最终判断")
    print("=" * 70)

    if not any_request_success:
        print("没有任何成功请求。请先检查 AK/SK、endpoint、网络。")
        return

    if real_files_found:
        print("结论：当前 key 可以看到真实 Flat Files 文件。")
        print("下一步应检查 prefix、日期、symbol 命名。")
    else:
        print("结论：当前 key 没有看到任何真实数据文件。")
        print("这通常表示以下情况之一：")
        print("1. 当前组织/订阅没有开通 Flat Files 数据访问")
        print("2. 当前 API key 不能访问 Flat Files 数据")
        print("3. 只能看到目录标记，看不到实际文件")
        print("4. 需要在控制台确认 Flat Files 产品是否已启用")
        

if __name__ == "__main__":
    main()
