
import hmac
import json
import logging
import os
import socket
import sys

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

            frame = json.loads(raw_body)

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
                    self.send_json(400, {"error": "Invalid message"})
                    return

                if set(message) != {"role", "content"}:
                    self.send_json(400, {"error": "Invalid message fields"})
                    return

                role = message["role"]
                content = message["content"]

                if role not in allowed_roles:
                    self.send_json(400, {"error": "Invalid message role"})
                    return

                if not isinstance(content, str):
                    self.send_json(400, {"error": "Invalid message content"})
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
                "Inference started: model=%s messages=%d bytes=%d tokens=%d",
                model,
                len(messages),
                len(provider_body),
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
                    "Provider HTTP error: status=%d body=%s",
                    exc.code,
                    error_body[:1000],
                )

                self.send_json(
                    503,
                    {"error": "Model provider request failed"},
                )
                return

            except (URLError, TimeoutError, socket.timeout) as exc:

                logger.error(
                    "Provider connection error: %s",
                    type(exc).__name__,
                )

                self.send_json(
                    503,
                    {"error": "Model provider unavailable"},
                )
                return

            # 8. Validate provider response size

            if len(raw) > 131072:

                logger.error("Provider response too large")

                self.send_json(
                    502,
                    {"error": "Provider response too large"},
                )
                return

            # 9. Parse OpenRouter response

            try:

                result = json.loads(raw)

                choices = result["choices"]

                if not isinstance(choices, list) or not choices:
                    raise ValueError("Missing choices")

                message = choices[0]["message"]
                response_text = message["content"]

                if (
                    not isinstance(response_text, str)
                    or not 0 < len(response_text.encode("utf-8")) <= 65536
                ):
                    raise ValueError("Invalid response content")

            except (ValueError, KeyError, TypeError, IndexError):

                logger.error("Invalid provider response structure")

                self.send_json(
                    502,
                    {"error": "Invalid model response"},
                )
                return

            # 10. Normalize usage

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

                if (
                    type(value) is int
                    and 0 <= value <= 1000000000
                ):
                    usage[target] = value

            if (
                "input_tokens" in usage
                and "output_tokens" in usage
            ):
                usage["total_tokens"] = (
                    usage["input_tokens"] + usage["output_tokens"]
                )

            # 11. Return competition-compatible response

            output = {
                "text": response_text,
                "usage": usage,
            }

            logger.info(
                "Inference completed: input=%s output=%s",
                usage.get("input_tokens", "unknown"),
                usage.get("output_tokens", "unknown"),
            )

            self.send_json(200, output)

        except (BrokenPipeError, ConnectionResetError):

            logger.warning("Client disconnected")

        except Exception as exc:

            logger.exception(
                "Unexpected gateway exception: %s",
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
