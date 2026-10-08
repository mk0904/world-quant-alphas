"""WorldQuant BRAIN API client. Auth, submit, poll, fetch alpha metrics."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

API_BASE = "https://api.worldquantbrain.com"


@dataclass(frozen=True)
class SimulationSettings:
    region: str = "USA"
    universe: str = "TOP1000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "NONE"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unit_handling: str = "VERIFY"
    nan_handling: str = "OFF"
    language: str = "FASTEXPR"
    instrument_type: str = "EQUITY"
    visualization: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": self.visualization,
        }


class WQAuthError(RuntimeError):
    pass


class WQBiometricError(WQAuthError):
    """Auth returned 200 but with `inquiry` payload — user must complete biometric/persona check."""

    def __init__(self, inquiry_url: str, raw: dict[str, Any]) -> None:
        super().__init__(
            f"biometric/persona check required. complete at: {inquiry_url}"
        )
        self.inquiry_url = inquiry_url
        self.raw = raw


# Backward-compat alias — the class was originally exported under the longer name.
WQBiometricRequired = WQBiometricError


class WQClient:
    def __init__(
        self,
        email: str,
        password: str,
        base_url: str = API_BASE,
        timeout_s: float = 30.0,
    ) -> None:
        if not email or not password:
            raise ValueError("email and password required")
        self._email = email
        self._password = password
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._session = requests.Session()
        self._authed = False

    def authenticate(self, *, max_retries: int = 4) -> None:
        url = f"{self._base}/authentication"
        for attempt in range(max_retries):
            resp = self._session.post(
                url,
                auth=HTTPBasicAuth(self._email, self._password),
                timeout=self._timeout,
            )
            if resp.status_code in (200, 201):
                try:
                    body = resp.json() if resp.text else {}
                except ValueError:
                    body = {}
                inquiry = body.get("inquiry") if isinstance(body, dict) else None
                if inquiry:
                    inquiry_id = (
                        inquiry.get("id") if isinstance(inquiry, dict) else str(inquiry)
                    )
                    inquiry_url = (
                        f"https://api.worldquantbrain.com/authentication/persona?inquiry={inquiry_id}"
                        if inquiry_id
                        else f"{self._base}/authentication/persona"
                    )
                    self._authed = False
                    raise WQBiometricRequired(inquiry_url, body)
                self._authed = True
                logger.info("auth ok")
                return
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2**attempt + 1.0))
                logger.warning("auth 429 rate limit; backoff %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                raise WQAuthError(
                    f"auth 401 — bad credentials or biometric/2FA required. body: {resp.text[:300]}"
                )
            raise WQAuthError(f"auth failed {resp.status_code}: {resp.text[:300]}")
        raise WQAuthError("auth failed: exceeded retries due to rate limits")

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> requests.Response:
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{self._base}{path_or_url}"
        )
        kwargs.setdefault("timeout", self._timeout)

        if not self._authed:
            self.authenticate()

        last_exc: Exception | None = None
        reauth_count = 0
        max_reauths = 1
        for attempt in range(max_retries):
            try:
                resp = self._session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning("net err %s; retry in %ss", exc, wait)
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                if reauth_count >= max_reauths:
                    raise WQAuthError(
                        f"persistent 401 on {method} {url} after {reauth_count} re-auth(s); "
                        f"body: {resp.text[:200]}"
                    )
                logger.warning("401; re-auth (count=%d)", reauth_count + 1)
                reauth_count += 1
                self._authed = False
                self.authenticate()
                continue

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2**attempt))
                logger.warning("429; backoff %ss", wait)
                time.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                wait = 2**attempt
                logger.warning("%s; retry in %ss", resp.status_code, wait)
                time.sleep(wait)
                continue

            return resp

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"exhausted retries on {method} {url}")

    def submit_simulation(
        self, expression: str, settings: SimulationSettings
    ) -> str:
        payload = {
            "type": "REGULAR",
            "settings": settings.to_payload(),
            "regular": expression,
        }
        resp = self._request("POST", "/simulations", json=payload)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"submit failed {resp.status_code}: {resp.text[:400]}"
            )
        location = resp.headers.get("Location")
        if not location:
            raise RuntimeError(
                f"no Location header on submit; body: {resp.text[:300]}"
            )
        return location

    def poll_simulation(
        self,
        location: str,
        *,
        timeout_s: int = 600,
        poll_floor_s: float = 2.0,
    ) -> dict[str, Any]:
        """Poll until terminal. 200 + no Retry-After = done. 202/204 or Retry-After = still running."""
        start = time.time()
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"sim poll exceeded {timeout_s}s; loc={location}"
                )
            resp = self._request("GET", location)
            if resp.status_code not in (200, 201, 202, 204):
                raise RuntimeError(
                    f"poll failed {resp.status_code}: {resp.text[:300]}"
                )
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(float(retry_after), poll_floor_s)
                except ValueError:
                    wait = poll_floor_s
                logger.info("running; sleep %ss", wait)
                time.sleep(wait)
                continue
            if resp.status_code in (202, 204):
                logger.info("status %d; sleep %ss", resp.status_code, poll_floor_s)
                time.sleep(poll_floor_s)
                continue
            return resp.json()

    def get_alpha(self, alpha_id: str) -> dict[str, Any]:
        resp = self._request("GET", f"/alphas/{alpha_id}")
        if resp.status_code != 200:
            raise RuntimeError(
                f"get alpha failed {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    def get_user_alphas(
        self, *, status: str | None = None, page_size: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch all alphas belonging to the authenticated user."""
        if page_size <= 0:
            raise ValueError("page_size must be greater than zero")

        params: dict[str, Any] = {"limit": page_size, "offset": 0}
        if status:
            params["status"] = status

        results: list[dict[str, Any]] = []
        while True:
            resp = self._request("GET", "/users/self/alphas", params=params)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"get user alphas failed {resp.status_code}: {resp.text[:300]}"
                )
            body = resp.json()
            if not isinstance(body, dict) or not isinstance(body.get("results"), list):
                raise RuntimeError("get user alphas returned an invalid response")

            page = body["results"]
            if any(not isinstance(alpha, dict) for alpha in page):
                raise RuntimeError("get user alphas returned an invalid alpha record")
            results.extend(page)

            params["offset"] += len(page)
            count = body.get("count")
            if count is not None:
                try:
                    if params["offset"] >= int(count):
                        break
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("get user alphas returned an invalid count") from exc
            elif len(page) < page_size:
                break
            if not page:
                break

        return results

    def get_alpha_pnl(self, alpha_id: str, *, max_retries: int = 4) -> dict[str, Any]:
        """Fetch PnL recordset for an alpha. Body: {records: [[date, pnl], ...], schema: ...}.

        WQ generates the recordset lazily — the first call after a sim completes
        sometimes returns 200 with an empty body. Retry with backoff on empty.
        """
        last_err = ""
        for attempt in range(max_retries):
            resp = self._request("GET", f"/alphas/{alpha_id}/recordsets/pnl")
            if resp.status_code != 200:
                raise RuntimeError(
                    f"get pnl failed {resp.status_code}: {resp.text[:300]}"
                )
            if resp.text.strip():
                return resp.json()
            wait = 2 ** attempt
            last_err = "empty body"
            logger.info("pnl empty for %s; retry in %ss", alpha_id, wait)
            time.sleep(wait)
        raise RuntimeError(
            f"pnl never populated for {alpha_id} after {max_retries} retries: {last_err}"
        )

    def get_data_fields(
        self,
        *,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        dataset_id: str | None = None,
        search: str | None = None,
        page_size: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch all data fields matching filters. Paginated via offset."""
        base_params: dict[str, Any] = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": page_size,
        }
        if dataset_id:
            base_params["dataset.id"] = dataset_id
        if search:
            base_params["search"] = search

        first = self._request("GET", "/data-fields", params=dict(base_params, offset=0))
        if first.status_code != 200:
            raise RuntimeError(
                f"data-fields failed {first.status_code}: {first.text[:300]}"
            )
        body = first.json()
        count = int(body.get("count", 0))
        results: list[dict[str, Any]] = list(body.get("results", []))
        logger.info("data-fields total=%d page0=%d", count, len(results))

        for offset in range(page_size, count, page_size):
            resp = self._request(
                "GET", "/data-fields", params=dict(base_params, offset=offset)
            )
            if resp.status_code != 200:
                logger.warning(
                    "data-fields page offset=%d failed %d", offset, resp.status_code
                )
                continue
            page = resp.json().get("results", [])
            results.extend(page)
            logger.info("data-fields offset=%d +%d total=%d", offset, len(page), len(results))

        return results

    def run_simulation(
        self, expression: str, settings: SimulationSettings
    ) -> dict[str, Any]:
        """Submit, poll to terminal, fetch alpha. Returns dict with sim + alpha."""
        location = self.submit_simulation(expression, settings)
        sim_result = self.poll_simulation(location)
        alpha_id = sim_result.get("alpha")
        alpha = self.get_alpha(alpha_id) if alpha_id else None
        return {
            "location": location,
            "simulation": sim_result,
            "alpha_id": alpha_id,
            "alpha": alpha,
        }
# client retry backoff optimized
