
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import (
    Request,
    build_opener,
    ProxyHandler,
    HTTPSHandler,
    HTTPRedirectHandler,
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GatewayHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        # Do not log request headers or credentials.
        pass

    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            expected_token = os.environ["TEAM_GATEWAY_TOKEN"]
            supplied_token = self.headers.get("Authorization", "")

            if self.path != "/gateway" or not hmac.compare_digest(
                supplied_token, "Bearer " + expected_token
            ):
                self.send_error(403)
                return

            size = int(self.headers.get("Content-Length", "0"))

            if not 0 < size <= 65536:
                self.send_error(400, "Invalid request size")
                return

            frame = json.loads(self.rfile.read(size))

            if set(frame) != {"messages", "max_output_tokens"}:
                self.send_error(400, "Invalid request fields")
                return

            messages = frame["messages"]
            max_tokens = frame["max_output_tokens"]

            if not isinstance(messages, list) or not 1 <= len(messages) <= 24:
                self.send_error(400, "Invalid messages")
                return

            if type(max_tokens) is not int or not 16 <= max_tokens <= 2048:
                self.send_error(400, "Invalid token limit")
                return

            provider_url = os.environ["TEAM_PROVIDER_URL"]

            if not provider_url.startswith("https://"):
                self.send_error(400, "HTTPS provider required")
                return

            payload = {
                "model": os.environ["TEAM_MODEL"],
                "messages": messages,
                "max_tokens": max_tokens,
            }

            request = Request(
                provider_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": (
                        "Bearer " + os.environ["TEAM_PROVIDER_KEY"]
                    ),
                },
            )

            opener = build_opener(
                ProxyHandler({}),
                HTTPSHandler(),
                NoRedirect(),
            )

            with opener.open(request, timeout=35) as response:
                raw = response.read(131073)

            if len(raw) > 131072:
                raise ValueError("Provider response too large")

            result = json.loads(raw)
            response_text = result["choices"][0]["message"]["content"]

            if (
                not isinstance(response_text, str)
                or not 0 < len(response_text.encode("utf-8")) <= 65536
            ):
                raise ValueError("Invalid model response")

            usage = result.get("usage", {})

            output = {
                "text": response_text,
                "usage": {
                    output_key: usage[input_key]
                    for input_key, output_key in [
                        ("prompt_tokens", "input_tokens"),
                        ("completion_tokens", "output_tokens"),
                    ]
                    if type(usage.get(input_key)) is int
                    and usage[input_key] >= 0
                },
            }

            self.send_json(200, output)

        except Exception:
            # Avoid exposing provider credentials or internal errors.
            self.send_error(503, "Team gateway unavailable")


if __name__ == "__main__":

    required = [
        "TEAM_PROVIDER_URL",
        "TEAM_PROVIDER_KEY",
        "TEAM_MODEL",
        "TEAM_GATEWAY_TOKEN",
    ]

    for key in required:
        if not os.environ.get(key):
            raise SystemExit("Missing environment variable: " + key)

    port = int(os.environ.get("PORT", "8080"))

    print(f"Workshop gateway starting on port {port}")

    HTTPServer(
        ("0.0.0.0", port),
        GatewayHandler,
    ).serve_forever()
