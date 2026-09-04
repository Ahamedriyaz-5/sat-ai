"""
Copernicus Data Space Ecosystem (CDSE) & Sentinel Hub Integration
Provides live Sentinel-1 (SAR Radar) and Sentinel-2 (Multispectral Optical) 
satellite queries, exact geolocation footprints, and live satellite image processing.
"""

import os
import time
import base64
import logging
import requests
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv, set_key

load_dotenv()

logger = logging.getLogger("satquery.api.copernicus")
router = APIRouter()

# ─────────────────────────────────────────────────────────────
# CDSE & Sentinel Hub Endpoints
# ─────────────────────────────────────────────────────────────
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

_CACHED_TOKEN: Optional[str] = None
_TOKEN_EXPIRY: float = 0.0


def get_copernicus_credentials() -> Dict[str, str]:
    """Retrieve Copernicus Data Space credentials from environment."""
    load_dotenv(override=True)
    client_id = os.environ.get("COPERNICUS_CLIENT_ID", "").strip()
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET", "").strip()
    api_key = os.environ.get("COPERNICUS_API_KEY", "").strip()
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "api_key": api_key,
    }


def get_access_token() -> Optional[str]:
    """
    Fetch an OAuth2 Bearer token from Copernicus Data Space Keycloak.
    Caches token until close to expiration (typically 60 min).
    """
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    creds = get_copernicus_credentials()
    
    # If direct token or API key provided
    if creds["api_key"] and not (creds["client_id"] and creds["client_secret"]):
        return creds["api_key"]

    if not creds["client_id"] or not creds["client_secret"]:
        return None

    # Check cache
    if _CACHED_TOKEN and time.time() < (_TOKEN_EXPIRY - 120):
        return _CACHED_TOKEN

    try:
        data = {
            "grant_type": "client_credentials",
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
        }
        res = requests.post(TOKEN_URL, data=data, timeout=12)
        if res.status_code == 200:
            payload = res.json()
            _CACHED_TOKEN = payload.get("access_token")
            expires_in = payload.get("expires_in", 3600)
            _TOKEN_EXPIRY = time.time() + float(expires_in)
            logger.info("Successfully refreshed Copernicus Data Space OAuth2 access token.")
            return _CACHED_TOKEN
        else:
            logger.warning(f"Failed to authenticate with Copernicus Data Space: {res.status_code} {res.text[:150]}")
            return None
    except Exception as exc:
        logger.error(f"Copernicus OAuth token exception: {exc}")
        return None


def query_copernicus_catalog(lat: float, lon: float, collection: str = "SENTINEL-2", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Query open Copernicus Data Space OData catalog for the latest satellite passes
    over the given coordinates without requiring credentials.
    """
    delta = 0.15
    min_lon, max_lon = lon - delta, lon + delta
    min_lat, max_lat = lat - delta, lat + delta
    aoi_wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"

    filter_query = (
        f"Collection/Name eq '{collection}' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')"
    )

    url = f"{CATALOG_ODATA_URL}?$filter={filter_query}&$orderby=ContentDate/Start desc&$top={limit}&$format=json"

    try:
        res = requests.get(url, timeout=14)
        if res.status_code == 200:
            data = res.json().get("value", [])
            results = []
            for item in data:
                name = item.get("Name", "")
                dates = item.get("ContentDate", {})
                start_dt = dates.get("Start", "")
                results.append({
                    "id": item.get("Id"),
                    "name": name,
                    "date": start_dt,
                    "collection": collection,
                    "satellite": "Sentinel-2 MSI" if collection == "SENTINEL-2" else "Sentinel-1 SAR",
                    "footprint": item.get("GeoFootprint"),
                })
            return results
    except Exception as err:
        logger.warning(f"Copernicus OData query error: {err}")
    return []


# ─────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────
class CredentialsConfig(BaseModel):
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    api_key: Optional[str] = None


class LiveSatelliteRequest(BaseModel):
    latitude: float
    longitude: float
    modality: str = "optical"  # "optical" (Sentinel-2) or "sar" (Sentinel-1)
    date: Optional[str] = None
    radius_km: float = 5.0


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@router.get("/copernicus/status")
async def get_status():
    """Check connection status to Copernicus Data Space Ecosystem."""
    creds = get_copernicus_credentials()
    has_keys = bool((creds["client_id"] and creds["client_secret"]) or creds["api_key"])
    token = get_access_token() if has_keys else None
    
    return {
        "status": "active" if token else ("configured" if has_keys else "unconfigured"),
        "has_credentials": has_keys,
        "token_valid": bool(token),
        "client_id": creds["client_id"][:6] + "..." if creds["client_id"] else None,
        "provider": "Copernicus Data Space Ecosystem (European Space Agency / Sentinel Hub)",
        "supported_missions": ["Sentinel-1 (SAR Radar)", "Sentinel-2 (Multispectral Optical)"],
        "help_url": "https://dataspace.copernicus.eu/",
    }


@router.post("/copernicus/configure")
async def configure_copernicus(payload: CredentialsConfig):
    """Save Copernicus Data Space API credentials to .env."""
    env_path = os.path.join(os.getcwd(), ".env")
    if payload.client_id:
        set_key(env_path, "COPERNICUS_CLIENT_ID", payload.client_id.strip())
        os.environ["COPERNICUS_CLIENT_ID"] = payload.client_id.strip()
    if payload.client_secret:
        set_key(env_path, "COPERNICUS_CLIENT_SECRET", payload.client_secret.strip())
        os.environ["COPERNICUS_CLIENT_SECRET"] = payload.client_secret.strip()
    if payload.api_key:
        set_key(env_path, "COPERNICUS_API_KEY", payload.api_key.strip())
        os.environ["COPERNICUS_API_KEY"] = payload.api_key.strip()

    # Reset cached token
    global _CACHED_TOKEN, _TOKEN_EXPIRY
    _CACHED_TOKEN = None
    _TOKEN_EXPIRY = 0.0

    token = get_access_token()
    return {
        "success": True,
        "message": "Copernicus Data Space credentials updated.",
        "connected": bool(token),
    }


@router.post("/copernicus/live-satellite")
async def get_live_satellite_pass(req: LiveSatelliteRequest):
    """
    Retrieve live Sentinel satellite pass data and imagery for the given coordinates
    via Copernicus Data Space Ecosystem.
    """
    lat, lon = req.latitude, req.longitude
    col = "SENTINEL-1" if req.modality.lower() == "sar" else "SENTINEL-2"

    # 1. Search recent acquisitions in Copernicus open catalog
    passes = query_copernicus_catalog(lat, lon, collection=col, limit=5)

    token = get_access_token()
    live_image_b64 = None
    source = "Copernicus Data Space Catalog"

    # 2. If OAuth token is available, request live Sentinel Hub Process API tile
    if token:
        try:
            delta = req.radius_km / 111.0
            bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

            if col == "SENTINEL-2":
                evalscript = """
                //VERSION=3
                function setup() {
                  return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                  };
                }
                function evaluatePixel(sample) {
                  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                }
                """
                payload = {
                    "input": {
                        "bounds": {
                            "bbox": bbox,
                            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                        },
                        "data": [{"type": "sentinel-2-l2a"}]
                    },
                    "output": {"width": 512, "height": 512, "responses": [{"format": {"type": "image/png"}}]},
                    "evalscript": evalscript
                }
            else:
                evalscript = """
                //VERSION=3
                function setup() {
                  return {
                    input: ["VV"],
                    output: { bands: 1 }
                  };
                }
                function evaluatePixel(sample) {
                  return [Math.max(0, Math.min(1, (sample.VV + 25) / 30))];
                }
                """
                payload = {
                    "input": {
                        "bounds": {
                            "bbox": bbox,
                            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                        },
                        "data": [{"type": "sentinel-1-grd"}]
                    },
                    "output": {"width": 512, "height": 512, "responses": [{"format": {"type": "image/png"}}]},
                    "evalscript": evalscript
                }

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            resp = requests.post(SH_PROCESS_URL, json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                live_image_b64 = base64.b64encode(resp.content).decode("utf-8")
                source = "Copernicus Sentinel Hub Process API"
            else:
                logger.warning(f"Sentinel Hub process error {resp.status_code}: {resp.text[:150]}")
        except Exception as err:
            logger.error(f"Live satellite process error: {err}")

    return {
        "status": "success",
        "latitude": lat,
        "longitude": lon,
        "modality": req.modality,
        "satellite": "Sentinel-2 MSI" if col == "SENTINEL-2" else "Sentinel-1 SAR",
        "live_image_b64": live_image_b64,
        "source": source,
        "recent_passes": passes,
        "authenticated": bool(token),
    }
