"""
osp_client.py — Python bridge between admet.py and the plumber.R OSP service.

Architecture:
  admet.py           →  build_drug_profile()  →  dict
  osp_client.py      →  _map_to_osp_params()  →  JSON POST /simulate
  plumber.R service  →  ospsuite-R + PK-Sim   →  C(t), Cmax, AUC

Usage:
    from engine.osp_client import OSPClient

    client = OSPClient()          # starts plumber.R subprocess
    result = client.simulate(
        drug_profile=build_drug_profile("Caffeine", ...),
        dose_mg=200,
        route="oral",
        t_end_h=24.0
    )
    print(result["nca"])          # {"cmax_mg_l": ..., "auc_mg_l_h": ..., ...}
    client.stop()
"""

import subprocess
import time
import os
import signal
import atexit
from pathlib import Path
from typing import Optional
import requests
import json
import logging

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
_DEFAULT_PORT    = 8000
_STARTUP_TIMEOUT = 30    # seconds to wait for plumber to be ready
_REQUEST_TIMEOUT = 120   # seconds per simulation call

# Path to plumber.R relative to this file
_PLUMBER_R = Path(__file__).parent / "osp_service" / "plumber.R"


class OSPClient:
    """
    Manages the plumber.R subprocess and exposes simulate() for Python callers.

    The client starts the R service on first use and keeps it alive for the
    duration of the Python process (or until .stop() is called).
    """

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        plumber_script: Optional[Path] = None,
        startup_timeout: int = _STARTUP_TIMEOUT,
    ):
        self.port    = port
        self.base    = f"http://127.0.0.1:{port}"
        self.script  = plumber_script or _PLUMBER_R
        self._proc: Optional[subprocess.Popen] = None
        self._timeout = startup_timeout

        if not self.script.exists():
            raise FileNotFoundError(f"plumber.R not found at: {self.script}")

        self._start()
        atexit.register(self.stop)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _start(self) -> None:
        """Launch the plumber.R REST service as a background subprocess."""
        cmd = [
            "Rscript", "-e",
            f"pr <- plumber::plumb('{self.script}'); "
            f"pr$run(port={self.port}, host='127.0.0.1')"
        ]

        env = os.environ.copy()
        # Ensure .NET is on PATH for rSharp/ospsuite
        dotnet_root = os.environ.get("DOTNET_ROOT", os.path.expanduser("~/.dotnet"))
        env["DOTNET_ROOT"] = dotnet_root
        env["PATH"]        = f"{dotnet_root}:{env.get('PATH', '')}"

        log.info("Starting plumber.R service on port %d ...", self.port)
        self._proc = subprocess.Popen(
            cmd,
            env    = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            preexec_fn = os.setsid,   # process group for clean kill
        )

        # Wait until the /health endpoint responds
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base}/health", timeout=2)
                if r.status_code == 200:
                    info = r.json()
                    log.info(
                        "OSP service ready — ospsuite %s, plumber %s",
                        info.get("ospsuite"), info.get("plumber")
                    )
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)

        # Timeout — collect stderr for diagnosis
        self._proc.poll()
        stderr = self._proc.stderr.read(2000).decode(errors="replace")
        raise RuntimeError(
            f"plumber.R did not start within {self._timeout}s.\n"
            f"stderr: {stderr}"
        )

    def stop(self) -> None:
        """Terminate the plumber.R subprocess."""
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=5)
                log.info("OSP service stopped.")
            except Exception as e:
                log.warning("Error stopping OSP service: %s", e)
        self._proc = None

    def health(self) -> dict:
        """Check if the service is alive."""
        r = requests.get(f"{self.base}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    # ── Parameter mapping ────────────────────────────────────────────────────

    @staticmethod
    def _map_to_osp_params(drug: dict, dose_mg: float, t_end_h: float) -> dict:
        """
        Translate an admet.py drug profile dict into the OSP bridge JSON schema.

        Handles:
        - pKa routing: acid/base/ampholyte → pka_acid / pka_base
        - Permeability: uses measured p_eff if available, else QSPR predicted
        - Rb: uses measured blood:plasma ratio if available

        Parameters
        ----------
        drug      : dict from build_drug_profile()
        dose_mg   : administered dose in mg
        t_end_h   : simulation duration in hours

        Returns
        -------
        dict ready to POST to /simulate as JSON
        """
        # ── pKa routing ─────────────────────────────────────────────────
        drug_type = drug.get("drug_type", "neutral")
        pka       = drug.get("pka") or 0.0

        if drug_type == "acidic":
            pka_acid, pka_base = float(pka), 0.0
        elif drug_type == "basic":
            pka_acid, pka_base = 0.0, float(pka)
        elif drug_type == "zwitterion":
            # admet.py stores pka as tuple/list for zwitterions
            pkas = pka if isinstance(pka, (list, tuple)) else [pka, 0.0]
            pka_acid = float(pkas[0]) if len(pkas) > 0 else 0.0
            pka_base = float(pkas[1]) if len(pkas) > 1 else 0.0
        else:
            pka_acid, pka_base = 0.0, 0.0

        # ── Permeability ─────────────────────────────────────────────────
        # Prefer measured human in vivo Peff (admet.py field: p_eff)
        # Fall back to QSPR-predicted value
        peff_cms = float(drug.get("p_eff") or drug.get("peff_cms") or 1e-5)

        # ── Blood:plasma ratio ────────────────────────────────────────────
        rb = float(drug.get("Rb") or drug.get("rb") or 1.0)

        return {
            "drug_name": str(drug.get("name", "Drug")),
            "logp":      float(drug.get("logp", 0.0)),
            "fup":       float(drug.get("fup",  0.5)),
            "mw":        float(drug.get("mw",   300.0)),
            "pka_acid":  pka_acid,
            "pka_base":  pka_base,
            "peff_cms":  peff_cms,
            "rb":        rb,
            "dose_mg":   float(dose_mg),
            "t_end_h":   float(t_end_h),
        }

    # ── Main entry point ─────────────────────────────────────────────────────

    def simulate(
        self,
        drug: dict,
        dose_mg: float,
        route: str   = "oral",
        t_end_h: float = 24.0,
    ) -> dict:
        """
        Run a PBPK simulation via the OSP service.

        Parameters
        ----------
        drug     : drug profile dict from admet.py build_drug_profile()
        dose_mg  : administered dose [mg]
        route    : "oral" or "iv"  (currently oral only — IV planned)
        t_end_h  : simulation duration [hours]

        Returns
        -------
        dict with keys:
            drug_name   str
            status      "success" | "error"
            nca         dict {cmax_mg_l, tmax_h, auc_mg_l_h, t_half_h}
            time_h      list[float]
            conc_mg_l   list[float]
            error       str  (only on failure)
        """
        if route not in ("oral", "iv"):
            raise ValueError(f"Unsupported route '{route}' — use 'oral' or 'iv'")

        payload = self._map_to_osp_params(drug, dose_mg, t_end_h)

        try:
            response = requests.post(
                f"{self.base}/simulate",
                json    = payload,
                timeout = _REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            return {
                "drug_name": payload["drug_name"],
                "status":    "error",
                "error":     f"Simulation timed out after {_REQUEST_TIMEOUT}s",
            }
        except requests.exceptions.HTTPError as e:
            body = {}
            try:
                body = e.response.json()
            except Exception:
                pass
            return {
                "drug_name": payload["drug_name"],
                "status":    "error",
                "error":     body.get("error", str(e)),
            }
        except Exception as e:
            return {
                "drug_name": payload["drug_name"],
                "status":    "error",
                "error":     str(e),
            }