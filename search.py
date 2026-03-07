import os
import time
import base64
import hashlib
import logging
import zlib

import requests
import streamlit as st
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from urllib.parse import urlencode

logger = logging.getLogger("Trinetra")


# ==================== DYNAMO CONTENT DECODER ====================

def decode_dynamo_content(item: dict) -> str:
    """
    Decode a DynamoDB text asset's content field.

    The Lambda write path may compress content before storing it.
    Always call this instead of reading item["content"] directly.

    Supported encodings:
        "plain"    — content is a plain UTF-8 string (default)
        "zlib+b64" — content is zlib-compressed, then base64-encoded
    """
    encoding = item.get("encoding", "plain")
    content  = item["content"]

    if encoding == "zlib+b64":
        return zlib.decompress(base64.b64decode(content)).decode("utf-8")

    return content


# ==================== WEB SEARCH ====================

class WebSearchEngine:

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def search(self, query: str, max_results: int = 8) -> list[dict]:
        try:
            params = {"q": query, "kl": "us-en", "kp": "-2"}
            url    = f"https://html.duckduckgo.com/html/?{urlencode(params)}"
            resp   = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            soup    = BeautifulSoup(resp.text, "html.parser")
            results = []

            for tag in soup.select(".result"):
                a    = tag.select_one(".result__a")
                snip = tag.select_one(".result__snippet")
                if not a:
                    continue
                title = a.get_text(strip=True)
                href  = a.get("href", "")
                if "uddg=" in href:
                    from urllib.parse import urlparse, parse_qs
                    parsed = parse_qs(urlparse(href).query)
                    href   = parsed.get("uddg", [href])[0]
                snippet = snip.get_text(strip=True) if snip else ""
                if title and href:
                    results.append({"title": title, "url": href, "snippet": snippet})
                if len(results) >= max_results:
                    break

            logger.info(f"WEB_SEARCH q={query!r} hits={len(results)}")
            return results
        except Exception as e:
            logger.error(f"Web search failed for {query!r}: {e}", exc_info=True)
            return []

    def fetch_page_text(self, url: str, max_chars: int = 3000) -> str:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text    = soup.get_text(separator="\n", strip=True)
            lines   = [l for l in text.splitlines() if l.strip()]
            cleaned = "\n".join(lines)
            return cleaned[:max_chars] + ("…" if len(cleaned) > max_chars else "")
        except Exception as e:
            logger.error(f"Page fetch failed for {url}: {e}", exc_info=True)
            return "Could not fetch page content."


# ==================== LAMBDA SEARCH CLIENT ====================

class LambdaSearchClient:
    """Calls trinetra_text_search Lambda via its Function URL."""

    def __init__(self):
        self.function_url = ""
        try:
            self.function_url = st.secrets.get("LAMBDA_SEARCH_URL", "")
        except Exception:
            self.function_url = os.getenv("LAMBDA_SEARCH_URL", "")

    def is_configured(self) -> bool:
        return bool(self.function_url)

    def search(self, query: str, modality: str = "all", limit: int = 10) -> dict:
        """
        Search via Lambda. Results containing text assets will have their
        content decoded via decode_dynamo_content() before being returned.
        """
        if not self.is_configured():
            return {"error": "LAMBDA_SEARCH_URL not set in secrets.toml"}

        retries = 3
        for attempt in range(retries):
            try:
                params = {"q": query, "modality": modality, "limit": limit}
                resp   = requests.get(self.function_url, params=params, timeout=10)

                if resp.status_code == 429:
                    # Lambda returned retryable throttle — back off and retry
                    if attempt < retries - 1:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    return {"error": "Search service is busy — try again shortly"}

                resp.raise_for_status()
                data = resp.json()

                # Decode any compressed text content before returning to caller
                for hit in data.get("results", []):
                    if hit.get("modality") == "text" and "content" in hit:
                        try:
                            hit["content"] = decode_dynamo_content(hit)
                        except Exception as e:
                            logger.warning(f"Failed to decode content for {hit.get('asset_id')}: {e}")

                return data

            except Exception as e:
                logger.error(f"Lambda search failed: {e}")
                return {"error": str(e)}

        return {"error": "Search service unavailable after retries"}


# ==================== TRINETRA INGEST CLIENT ====================

class TrinetraIngestClient:
    """
    Sends assets to the Trinetra Lambda ingestion endpoint.

    Supports text now; structured for image/audio expansion later
    (each modality will have its own Lambda endpoint and method).

    Configure in secrets.toml:
        LAMBDA_INGEST_URL = "https://<your-function-url>"
    """

    def __init__(self):
        self.ingest_url = ""
        try:
            self.ingest_url = st.secrets.get("LAMBDA_INGEST_URL", "")
        except Exception:
            self.ingest_url = os.getenv("LAMBDA_INGEST_URL", "")

    def is_configured(self) -> bool:
        return bool(self.ingest_url)

    def _is_duplicate(self, text: str) -> bool:
        """
        Check DynamoDB (via Lambda) whether this content hash already exists.
        Falls back to False (allow write) if the check fails — better to store
        a duplicate than to silently drop content.

        In-memory sets are intentionally avoided: they don't survive restarts
        and don't work across multiple Streamlit instances.
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not self.is_configured():
            return False
        try:
            params = {"content_hash": content_hash}
            resp   = requests.get(self.ingest_url, params=params, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("exists", False)
        except Exception as e:
            logger.warning(f"Dedup check failed — allowing write: {e}")
        return False

    def send_text(self, text: str, language: str = "en") -> dict:
        """
        Ingest a text asset. Retries on 429. Returns the Lambda response dict.
        Skips the write if the content hash already exists in DynamoDB.
        """
        if not self.is_configured():
            return {"error": "LAMBDA_INGEST_URL not set in secrets.toml"}

        if self._is_duplicate(text):
            logger.info("INGEST_SKIPPED duplicate content hash")
            return {"skipped": True, "reason": "duplicate"}

        retries = 3
        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.ingest_url,
                    json={"text": text, "language": language},
                    timeout=10,
                )

                if resp.status_code == 429:
                    if attempt < retries - 1:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    return {"error": "Ingest service is busy — try again shortly"}

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                logger.error(f"Text ingest failed: {e}")
                return {"error": str(e)}

        return {"error": "Ingest service unavailable after retries"}

    # Future modality methods follow the same pattern:
    #
    # def send_image(self, image_path: str, ...) -> dict:
    #     ...POST to LAMBDA_INGEST_IMAGE_URL...
    #
    # def send_audio(self, audio_path: str, ...) -> dict:
    #     ...POST to LAMBDA_INGEST_AUDIO_URL...


# ==================== AWS REVERSE SEARCH ====================

class AWSReverseSearchEngine:
    """AWS-powered reverse search using Rekognition."""

    def __init__(self):
        import boto3

        self.aws_access_key = ""
        self.aws_secret_key = ""
        self.aws_region     = "us-east-1"

        try:
            if "AWS_ACCESS_KEY_ID" in st.secrets:
                self.aws_access_key = str(st.secrets["AWS_ACCESS_KEY_ID"]).strip()
                self.aws_secret_key = str(st.secrets["AWS_SECRET_ACCESS_KEY"]).strip()
                self.aws_region     = str(st.secrets.get("AWS_REGION", "us-east-1")).strip()
        except Exception:
            pass

        if not self.aws_access_key:
            self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
            self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
            self.aws_region     = os.getenv("AWS_REGION", "us-east-1").strip()

        if self.is_configured():
            try:
                self.rekognition = boto3.client(
                    "rekognition",
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region,
                )
                logger.info("AWS Rekognition initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AWS: {e}")

    def is_configured(self) -> bool:
        return bool(self.aws_access_key) and bool(self.aws_secret_key)

    def test_aws_connection(self):
        if not self.is_configured():
            return False, "❌ AWS not configured. Add credentials to secrets.toml"
        try:
            self.rekognition.detect_labels(
                Image={"Bytes": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100}, MaxLabels=1
            )
            return False, "Test image invalid (expected)"
        except self.rekognition.exceptions.InvalidImageFormatException:
            return True, f"✅ AWS Rekognition connected! Region: {self.aws_region}"
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msgs = {
                "InvalidClientTokenId": "❌ Invalid AWS_ACCESS_KEY_ID",
                "SignatureDoesNotMatch": "❌ Invalid AWS_SECRET_ACCESS_KEY",
            }
            return False, msgs.get(code, f"❌ AWS error: {code}")
        except Exception as e:
            return False, f"❌ Error: {str(e)}"

    def analyze_image_with_rekognition(self, image_path: str) -> dict:
        if not self.is_configured():
            return {"error": "AWS not configured"}
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            results = {"labels": [], "text": [], "celebrities": [], "faces": 0}

            try:
                label_response = self.rekognition.detect_labels(
                    Image={"Bytes": image_bytes}, MaxLabels=20, MinConfidence=70
                )
                results["labels"] = [
                    {"name": lb["Name"], "confidence": lb["Confidence"]}
                    for lb in label_response["Labels"]
                ]
            except Exception as e:
                logger.warning(f"Label detection failed: {e}")

            try:
                text_response = self.rekognition.detect_text(Image={"Bytes": image_bytes})
                results["text"] = [
                    t["DetectedText"]
                    for t in text_response["TextDetections"]
                    if t["Type"] == "LINE" and t["Confidence"] > 80
                ]
            except Exception as e:
                logger.warning(f"Text detection failed: {e}")

            try:
                face_response  = self.rekognition.detect_faces(
                    Image={"Bytes": image_bytes}, Attributes=["ALL"]
                )
                results["faces"] = len(face_response["FaceDetails"])
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")

            return results
        except Exception as e:
            logger.error(f"Rekognition failed: {e}", exc_info=True)
            return {"error": str(e)}

    def search_web_from_image_analysis(self, image_path: str, web_search_engine) -> dict:
        analysis = self.analyze_image_with_rekognition(image_path)
        if "error" in analysis:
            return {"error": analysis["error"]}

        search_queries = []
        if analysis["labels"]:
            top_labels = [lb["name"] for lb in analysis["labels"][:5]]
            search_queries.append(" ".join(top_labels))
        if analysis["text"]:
            search_queries.append(" ".join(analysis["text"][:3]))
        if not search_queries:
            search_queries = ["image analysis"]

        all_results = []
        for query in search_queries[:2]:
            all_results.extend(web_search_engine.search(query, max_results=5))

        seen_urls      = set()
        unique_results = []
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)

        return {
            "analysis":       analysis,
            "web_results":    unique_results[:10],
            "search_queries": search_queries,
        }

    def estimate_costs(self, num_images: int = 0) -> dict:
        costs = {"rekognition": num_images * 0.001, "total": num_images * 0.001}
        return {
            "costs":            costs,
            "remaining_credit": 100 - costs["total"],
            "can_process":      {"images": int((100 - costs["total"]) / 0.001)},
        }