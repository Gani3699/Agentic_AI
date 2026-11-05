"""
Workshop Parts Desk LLM Gateway
Implements the competition model relay protocol for OpenRouter/OpenAI-compatible backends.
"""

import hmac
import json
import logging
import os
import socket
import sys
import time
import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import (
    Request,
    build_opener,
    ProxyHandler,
    HTTPSHandler,
    HTTPRedirectHandler,
)
from urllib.error import HTTPError, URLError


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("workshop_gateway")


# --------------------------------------------------
# Disable redirects
# --------------------------------------------------

class NoRedirect(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# --------------------------------------------------
# Response Extraction Helper
# --------------------------------------------------

def extract_model_output(result, logger_instance=logger, req_id=""):
    """
    Extracts (response_text, usage, error_reason) from an OpenRouter/OpenAI-compatible result dictionary.
    Returns:
        response_text: Valid nonempty string or None
        usage: Normalized usage dict with integer input_tokens, output_tokens, total_tokens
        error_reason: None if successful, or string describing why extraction failed
    """
    if not isinstance(result, dict):
        if logger_instance:
            logger_instance.error("[%s] Provider JSON is not an object: type=%s", req_id, type(result).__name__)
        return None, {}, "Invalid provider response structure"

    # Check if provider returned an error payload inside the body
    if "error" in result:
        err = result.get("error")
        err_code = err.get("code") if isinstance(err, dict) else None
        err_msg = (
            err.get("message", "unknown")
            if isinstance(err, dict)
            else str(err)
        )
        if logger_instance:
            logger_instance.error(
                "[%s] Provider error in payload: code=%s message=%s",
                req_id,
                err_code,
                err_msg[:200],
            )
        return None, {}, "Model provider returned an error"

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        if logger_instance:
            logger_instance.error(
                "[%s] Missing or empty choices in provider response: top_keys=%s",
                req_id,
                list(result.keys()),
            )
        return None, {}, "Missing choices in model response"

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        if logger_instance:
            logger_instance.error(
                "[%s] First choice is not an object: type=%s",
                req_id,
                type(first_choice).__name__,
            )
        return None, {}, "Invalid choice structure in model response"

    finish_reason = first_choice.get("finish_reason")
    message = first_choice.get("message")

    response_text = None
    if not isinstance(message, dict):
        legacy_text = first_choice.get("text")
        if isinstance(legacy_text, str) and legacy_text.strip():
            response_text = legacy_text.strip()
        else:
            if logger_instance:
                logger_instance.error(
                    "[%s] Choice missing message object: choice_keys=%s finish_reason=%s",
                    req_id,
                    list(first_choice.keys()),
                    finish_reason,
                )
            return None, {}, "Missing message in model choice"
    else:
        raw_content = message.get("content")
        has_reasoning = bool(message.get("reasoning")) or ("reasoning" in message)
        message_keys = list(message.keys())

        if isinstance(raw_content, str):
            cleaned = raw_content.strip()
            if cleaned:
                response_text = cleaned
        elif isinstance(raw_content, list):
            # Handle array of content blocks e.g. [{"type": "text", "text": "..."}]
            text_parts = []
            for block in raw_content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
                    elif "text" in block and isinstance(block["text"], str):
                        text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            combined = "".join(text_parts).strip()
            if combined:
                response_text = combined

        # Support reasoning text fallback for reasoning models (e.g. DeepSeek v4.1 Flash)
        # where content may be null/empty while reasoning contains the generated response.
        if not response_text and isinstance(message.get("reasoning"), str):
            cleaned_reasoning = message["reasoning"].strip()
            if cleaned_reasoning:
                if logger_instance:
                    logger_instance.info("[%s] Using reasoning fallback for model response text", req_id)
                response_text = cleaned_reasoning

        if not response_text:
            if logger_instance:
                logger_instance.error(
                    "[%s] No valid text in message: top_keys=%s msg_keys=%s content_type=%s finish_reason=%s has_reasoning=%s",
                    req_id,
                    list(result.keys()),
                    message_keys,
                    type(raw_content).__name__,
                    finish_reason,
                    has_reasoning,
                )
            return None, {}, "No valid model response text"

    # Enforce maximum text cap (65,536 bytes) safely
    if response_text:
        encoded = response_text.encode("utf-8")
        if len(encoded) > 65536:
            response_text = encoded[:65000].decode("utf-8", errors="ignore")

    # Normalize usage
    provider_usage = result.get("usage", {})
    if not isinstance(provider_usage, dict):
        provider_usage = {}

    usage = {}
    mapping = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
    }

    for source, target in mapping.items():
        value = provider_usage.get(source)
        if type(value) is int and 0 <= value <= 1000000000:
            usage[target] = value

    if "input_tokens" in usage and "output_tokens" in usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    elif "total_tokens" in provider_usage and type(provider_usage["total_tokens"]) is int:
        usage["total_tokens"] = provider_usage["total_tokens"]

    return response_text, usage, None


# --------------------------------------------------
# Gateway handler
# --------------------------------------------------

class GatewayHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        # Never log request headers or credentials.
        pass

    def send_json(self, status, data):

        body = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()

        self.wfile.write(body)
        self.close_connection = True

    # --------------------------------------------------
    # Health check
    # --------------------------------------------------

    def do_GET(self):

        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, {"error": "Not found"})

    # --------------------------------------------------
    # Gateway endpoint
    # --------------------------------------------------

    def do_POST(self):
        req_id = uuid.uuid4().hex[:8]
        start_time = time.monotonic()

        try:

            # 1. Authentication

            expected_token = os.environ["TEAM_GATEWAY_TOKEN"]

            supplied_token = self.headers.get(
                "Authorization", ""
            )

            if self.path != "/gateway":
                self.send_json(404, {"error": "Not found"})
                return

            if not hmac.compare_digest(
                supplied_token,
                "Bearer " + expected_token,
            ):
                logger.warning("[%s] 403 Forbidden: Invalid bearer token supplied", req_id)
                self.send_json(403, {"error": "Forbidden"})
                return

            # 2. Request size

            content_length = self.headers.get("Content-Length")

            if content_length is None:
                self.send_json(400, {"error": "Missing Content-Length"})
                return

            try:
                size = int(content_length)
            except ValueError:
                self.send_json(400, {"error": "Invalid Content-Length"})
                return

            if not 0 < size <= 65536:
                self.send_json(400, {"error": "Invalid request size"})
                return

            raw_body = self.rfile.read(size)

            if len(raw_body) != size:
                self.send_json(400, {"error": "Incomplete request"})
                return

            # 3. Parse request

            try:
                frame = json.loads(raw_body)
            except Exception:
                self.send_json(400, {"error": "Invalid JSON in request"})
                return

            if not isinstance(frame, dict):
                self.send_json(400, {"error": "Invalid JSON object"})
                return

            if set(frame) != {"messages", "max_output_tokens"}:
                self.send_json(400, {"error": "Invalid request fields"})
                return

            messages = frame["messages"]
            max_tokens = frame["max_output_tokens"]

            if (
                not isinstance(messages, list)
                or not 1 <= len(messages) <= 24
            ):
                self.send_json(400, {"error": "Invalid messages"})
                return

            # 4. Validate message structure

            allowed_roles = {"system", "user", "assistant"}
            has_non_system = False

            for message in messages:

                if not isinstance(message, dict):
                    self.send_json(400, {"error": "Invalid message structure"})
                    return

                if set(message) != {"role", "content"}:
                    self.send_json(400, {"error": "Invalid message fields"})
                    return

                role = message["role"]
                content = message["content"]

                if role not in allowed_roles:
                    self.send_json(400, {"error": "Invalid role"})
                    return

                if not isinstance(content, str) or not content:
                    self.send_json(400, {"error": "Empty message content"})
                    return

                if len(content.encode("utf-8")) > 16384:
                    self.send_json(400, {"error": "Message content too large"})
                    return

                if role != "system":
                    has_non_system = True

            if not has_non_system:
                self.send_json(400, {"error": "Missing non-system message"})
                return

            if (
                type(max_tokens) is not int
                or not 16 <= max_tokens <= 2048
            ):
                self.send_json(400, {"error": "Invalid token limit"})
                return

            # 5. Provider configuration

            provider_url = os.environ["TEAM_PROVIDER_URL"]
            provider_key = os.environ["TEAM_PROVIDER_KEY"]
            model = os.environ["TEAM_MODEL"]

            if not provider_url.startswith("https://"):
                self.send_json(500, {"error": "Invalid provider URL"})
                return

            # 6. Build OpenRouter request

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            provider_body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")

            request = Request(
                provider_url,
                data=provider_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": "Bearer " + provider_key,
                },
                method="POST",
            )

            opener = build_opener(
                ProxyHandler({}),
                HTTPSHandler(),
                NoRedirect(),
            )

            logger.info(
                "[%s] Calling provider: model=%s max_tokens=%d",
                req_id,
                model,
                max_tokens,
            )

            # 7. Call provider

            try:

                with opener.open(
                    request,
                    timeout=35,
                ) as response:

                    raw = response.read(131073)

            except HTTPError as exc:

                # Read provider error without exposing credentials.
                error_body = exc.read(2048).decode(
                    "utf-8",
                    errors="replace",
                )

                logger.error(
                    "[%s] Provider HTTP error: duration=%.2fs status=%d body=%s",
                    req_id,
                    time.monotonic() - start_time,
                    exc.code,
                    error_body[:500],
                )

                self.send_json(
                    503,
                    {"error": "Model provider request failed"},
                )
                return

            except (URLError, TimeoutError, socket.timeout) as exc:

                logger.error(
                    "[%s] Provider connection error: duration=%.2fs type=%s",
                    req_id,
                    time.monotonic() - start_time,
                    type(exc).__name__,
                )

                self.send_json(
                    503,
                    {"error": "Model provider unavailable"},
                )
                return

            # 8. Validate provider response size

            if len(raw) > 131072:

                logger.error("[%s] Provider response too large: bytes=%d", req_id, len(raw))

                self.send_json(
                    502,
                    {"error": "Provider response too large"},
                )
                return

            # 9. Parse OpenRouter response

            try:
                result = json.loads(raw)
            except Exception:
                logger.error("[%s] Failed to parse provider response as JSON", req_id)
                self.send_json(
                    502,
                    {"error": "Invalid JSON from model provider"},
                )
                return

            response_text, usage, error_reason = extract_model_output(
                result,
                logger_instance=logger,
                req_id=req_id,
            )

            if error_reason:
                self.send_json(502, {"error": error_reason})
                return

            # 10. Return competition-compatible response

            output = {
                "text": response_text,
                "usage": usage,
            }

            logger.info(
                "[%s] Inference completed: model=%s duration=%.2fs input=%s output=%s",
                req_id,
                model,
                time.monotonic() - start_time,
                usage.get("input_tokens", "unknown"),
                usage.get("output_tokens", "unknown"),
            )

            self.send_json(200, output)

        except (BrokenPipeError, ConnectionResetError):

            logger.warning("[%s] Client disconnected", req_id)

        except Exception as exc:

            logger.exception(
                "[%s] Unexpected gateway exception: duration=%.2fs type=%s",
                req_id,
                time.monotonic() - start_time,
                type(exc).__name__,
            )

            try:
                self.send_json(
                    503,
                    {"error": "Team gateway unavailable"},
                )
            except (BrokenPipeError, ConnectionResetError):
                pass


# --------------------------------------------------
# Server startup
# --------------------------------------------------

if __name__ == "__main__":

    required = [
        "TEAM_PROVIDER_URL",
        "TEAM_PROVIDER_KEY",
        "TEAM_MODEL",
        "TEAM_GATEWAY_TOKEN",
    ]

    for key in required:

        if not os.environ.get(key):
            raise SystemExit(
                "Missing environment variable: " + key
            )

    port = int(os.environ.get("PORT", "8080"))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        GatewayHandler,
    )

    logger.info("Workshop gateway starting on port %d", port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Gateway shutting down")
    finally:
        server.server_close()
