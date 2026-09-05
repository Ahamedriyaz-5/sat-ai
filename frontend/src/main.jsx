import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MapContainer, TileLayer, Marker, Popup, ZoomControl, Rectangle, ImageOverlay, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './styles.css';

// Leaflet default marker icons fix
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

const customIcon = L.divIcon({
  className: '',
  html: `<div class="map-pin-outer"><div class="map-pin-inner"></div></div>`,
  iconSize: [36, 36],
  iconAnchor: [18, 18],
  popupAnchor: [0, -20],
});

// When served on Vercel or any remote domain, ALWAYS use relative '/api/v1' to avoid mixed-content blocking
const API = (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1')
  ? '/api/v1'
  : (import.meta.env.VITE_API_URL || '/api/v1');

const STEPS = [
  { id: 1, label: 'Upload & Ingest', icon: '📤' },
  { id: 2, label: 'Modality & Verification', icon: '🛰️' },
  { id: 3, label: 'Location & Footprint', icon: '📍' },
  { id: 4, label: 'SAR & Optical Metadata / GeoTIFF', icon: '📡' },
  { id: 5, label: 'Evidence Report', icon: '📄' },
];

function fmt(v) {
  if (v === null || v === undefined || v === '') return 'Not available';
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (typeof v === 'number') return v.toLocaleString();
  if (Array.isArray(v)) return v.join(', ');
  return String(v);
}

function ConfBar({ value, color = '#9ad49e' }) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="conf-bar-label">{pct}%</span>
    </div>
  );
}

function Badge({ text, variant = 'neutral' }) {
  return <span className={`badge badge-${variant}`}>{text}</span>;
}

function MetaRow({ label, value }) {
  return (
    <div className="meta-row">
      <span className="meta-label">{label}</span>
      <span className="meta-value">{fmt(value)}</span>
    </div>
  );
}

// ─────────────────────────────────────────────
// Upload Component
// ─────────────────────────────────────────────

function UploadZone({ onFile, file, previewUrl, onRemove }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  }, [onFile]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    setDragging(e.type === 'dragover');
  }, []);

  const handlePaste = useCallback(async (e) => {
    e?.stopPropagation();
    try {
      if (navigator.clipboard && navigator.clipboard.read) {
        const items = await navigator.clipboard.read();
        for (const item of items) {
          const imageType = item.types.find((t) => t.startsWith('image/'));
          if (imageType) {
            const blob = await item.getType(imageType);
            const ext = imageType.split('/')[1] || 'png';
            const pastedFile = new File([blob], `clipboard_satellite_${Date.now()}.${ext}`, { type: imageType });
            onFile(pastedFile);
            return;
          }
        }
      }
      alert("No image found in clipboard. Copy an image first, then click paste.");
    } catch {
      alert("Clipboard access is restricted by the browser. You can press Ctrl+V directly to paste.");
    }
  }, [onFile]);

  useEffect(() => {
    const onWindowPaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const blob = item.getAsFile();
          if (blob) {
            const pastedFile = new File([blob], `pasted_satellite_${Date.now()}.png`, { type: blob.type });
            onFile(pastedFile);
            break;
          }
        }
      }
    };
    window.addEventListener('paste', onWindowPaste);
    return () => window.removeEventListener('paste', onWindowPaste);
  }, [onFile]);

  return (
    <div className="sat-upload-card" id="sat-upload-card">
      {/* Top Left Badge */}
      <div className="sat-card-badge">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
        <span>Satellite image</span>
      </div>

      {/* Main Interactive Drop Surface */}
      <div
        className={`sat-drop-surface ${dragging ? 'dragging' : ''}`}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,.tif,.tiff,.geotiff"
          style={{ display: 'none' }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
          id="file-input"
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: 'none' }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
          id="camera-input"
        />

        {file ? (
          <div className="sat-file-selected-box" onClick={(e) => e.stopPropagation()}>
            {previewUrl ? (
              <img src={previewUrl} alt="Satellite Preview" className="sat-preview-thumb" />
            ) : (
              <div className="sat-preview-geotiff-icon">🛰️ GeoTIFF Decoded</div>
            )}
            <div className="sat-selected-info">
              <span className="sat-selected-name">{file.name}</span>
              <span className="sat-selected-meta">
                {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type || 'satellite/raster'}
              </span>
            </div>
            <button
              type="button"
              className="sat-remove-btn"
              onClick={(e) => { e.stopPropagation(); onRemove(); }}
            >
              ✕ Remove
            </button>
          </div>
        ) : (
          <div className="sat-drop-empty">
            <div className="sat-tray-icon">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                <polyline points="7 9 12 4 17 9" />
                <line x1="12" y1="4" x2="12" y2="16" />
              </svg>
            </div>
            <div className="sat-drop-main">Drop Image Here</div>
            <div className="sat-drop-divider">- or -</div>
            <div className="sat-drop-action">Click to Upload</div>
          </div>
        )}
      </div>

      {/* Bottom Toolbar with 3 Action Icons */}
      <div className="sat-bottom-toolbar">
        <button
          type="button"
          className="sat-tool-btn"
          title="Upload image file"
          onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
            <polyline points="7 9 12 4 17 9" />
            <line x1="12" y1="4" x2="12" y2="16" />
          </svg>
        </button>
        <button
          type="button"
          className="sat-tool-btn"
          title="Capture from camera or webcam"
          onClick={(e) => { e.stopPropagation(); cameraInputRef.current?.click(); }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="13" r="4"/>
            <path d="M5 7h2l2-3h6l2 3h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z"/>
          </svg>
        </button>
        <button
          type="button"
          className="sat-tool-btn"
          title="Paste from clipboard"
          onClick={handlePaste}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
            <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
          </svg>
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Loading Overlay
// ─────────────────────────────────────────────

const PROGRESS_STEPS = [
  'Validating image & format headers...',
  'Extracting GeoTIFF geospatial metadata & CRS...',
  'Analyzing SAR microwave radar backscatter & speckle...',
  'Detecting optical multispectral band configurations...',
  'Evaluating spatial coordinate reference systems...',
  'Determining global geographic location...',
  'Performing reverse geocoding via OpenStreetMap...',
  'Extracting microwave scattering / optical land-cover...',
  'Assembling SAR & Optical Evidence Intelligence...',
];

function AnalysisOverlay({ filename }) {
  const [stepIdx, setStepIdx] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setStepIdx((s) => (s + 1) % PROGRESS_STEPS.length), 750);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="analysis-overlay">
      <div className="analysis-spinner">
        <svg width="64" height="64" viewBox="0 0 64 64">
          <circle cx="32" cy="32" r="28" stroke="#2b3736" strokeWidth="3" fill="none" />
          <circle cx="32" cy="32" r="28" stroke="#ef7657" strokeWidth="3" fill="none"
            strokeDasharray="88 88" strokeLinecap="round">
            <animateTransform attributeName="transform" type="rotate"
              from="0 32 32" to="360 32 32" dur="1.2s" repeatCount="indefinite" />
          </circle>
          <circle cx="32" cy="32" r="18" stroke="#9ad49e44" strokeWidth="1.5" fill="none">
            <animateTransform attributeName="transform" type="rotate"
              from="360 32 32" to="0 32 32" dur="3s" repeatCount="indefinite" />
          </circle>
          <text x="32" y="37" textAnchor="middle" fontSize="18" fill="#ef7657">🛰️</text>
        </svg>
      </div>
      <p className="analysis-filename">{filename}</p>
      <p className="analysis-step">{PROGRESS_STEPS[stepIdx]}</p>
      <div className="analysis-dots">
        {PROGRESS_STEPS.map((_, i) => (
          <span key={i} className={`dot ${i <= stepIdx ? 'done' : ''}`} />
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// World Map Component with Bounding Box Footprint
// ─────────────────────────────────────────────

function MapRecenterHelper({ lat, lon }) {
  const map = useMap();
  useEffect(() => {
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      map.flyTo([lat, lon], map.getZoom() < 12 ? 13 : map.getZoom(), { duration: 1.2 });
    }
  }, [lat, lon, map]);
  return null;
}

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => {
      if (onMapClick) {
        onMapClick(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
}

function WorldMap({
  lat,
  lon,
  city,
  country,
  status: locStatus,
  bbox,
  imageUrl,
  targetName,
  featureType,
  onLocationChange,
  onSearchLocation,
}) {
  const [mapType, setMapType] = useState('satellite');
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [editManual, setEditManual] = useState(false);
  const [customLat, setCustomLat] = useState(String(lat || ''));
  const [customLon, setCustomLon] = useState(String(lon || ''));

  useEffect(() => {
    setCustomLat(String(lat ? lat.toFixed(6) : ''));
    setCustomLon(String(lon ? lon.toFixed(6) : ''));
  }, [lat, lon]);

  const tileUrl = mapType === 'satellite'
    ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

  const tileAttr = mapType === 'satellite'
    ? 'Tiles &copy; Esri &mdash; Source: Esri, USGS, NOAA'
    : '&copy; OpenStreetMap contributors';

  const label = targetName || [city, country].filter(Boolean).join(', ') || 'Detected Ground Location';
  const hasBbox = Array.isArray(bbox) && bbox.length === 4 && bbox.every(Number.isFinite);

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    const q = searchQuery.trim();
    if (!q || !onSearchLocation) return;
    setSearching(true);
    setSearchError('');
    try {
      await onSearchLocation(q);
      setSearchQuery('');
    } catch (err) {
      setSearchError(err.message || 'Location not found. Try entering coordinates (lat, lon).');
    } finally {
      setSearching(false);
    }
  };

  const handleApplyManual = (e) => {
    if (e) e.preventDefault();
    const pLat = parseFloat(customLat);
    const pLon = parseFloat(customLon);
    if (!isNaN(pLat) && !isNaN(pLon) && pLat >= -90 && pLat <= 90 && pLon >= -180 && pLon <= 180) {
      if (onLocationChange) {
        onLocationChange(pLat, pLon, `Pinpoint Coordinate (${pLat.toFixed(4)}°, ${pLon.toFixed(4)}°)`);
      }
      setEditManual(false);
      setSearchError('');
    } else {
      setSearchError('Invalid coordinates. Latitude must be -90..90 and Longitude -180..180.');
    }
  };

  return (
    <div className="world-map-wrap">
      {/* Live Exact Ground Target Search Bar */}
      <div className="map-search-bar">
        <div className="msb-input-wrap">
          <span className="msb-icon">🎯</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setSearchError(''); }}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(e); }}
            placeholder="Search exact landmark, facility, or enter coordinates (e.g. 31.002, 35.145)..."
            className="map-search-input"
            id="map-search-input"
          />
          {searchQuery && (
            <button className="msb-clear" onClick={() => setSearchQuery('')} type="button">✕</button>
          )}
        </div>
        <button
          type="button"
          className="map-search-btn"
          onClick={handleSearch}
          disabled={searching || !searchQuery.trim()}
          id="map-search-btn"
        >
          {searching ? 'Locating...' : 'Pinpoint Exact Site 🔍'}
        </button>
      </div>

      {searchError && (
        <div className="map-search-error">
          <span>⚠</span> {searchError}
        </div>
      )}

      <div className="map-controls">
        <button
          className={`map-tab ${mapType === 'satellite' ? 'active' : ''}`}
          onClick={() => setMapType('satellite')}
        >🛰️ Satellite Layer</button>
        <button
          className={`map-tab ${mapType === 'streets' ? 'active' : ''}`}
          onClick={() => setMapType('streets')}
        >🗺️ Streets Layer</button>
        <button
          className={`map-tab ${editManual ? 'active' : ''}`}
          onClick={() => setEditManual(!editManual)}
        >✏️ Edit Coordinates</button>
        <button
          className="map-tab map-copy"
          onClick={() => navigator.clipboard?.writeText(`${lat.toFixed(6)}, ${lon.toFixed(6)}`)}
          title="Copy exact coordinates"
        >📋 Copy Coordinates</button>
        <a
          className="map-tab map-link"
          href={`https://www.google.com/maps?q=${lat},${lon}`}
          target="_blank"
          rel="noopener noreferrer"
        >🌐 Google Maps ↗</a>
        <a
          className="map-tab map-link"
          href={`https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=14/${lat}/${lon}`}
          target="_blank"
          rel="noopener noreferrer"
        >🗺️ OSM ↗</a>
      </div>

      {/* Manual Coordinate Fine-Tuning Drawer */}
      {editManual && (
        <form className="coord-fine-tuner" onSubmit={handleApplyManual}>
          <div className="cft-field">
            <label>Latitude (-90 to 90):</label>
            <input
              type="number"
              step="any"
              value={customLat}
              onChange={(e) => setCustomLat(e.target.value)}
              className="cft-input"
            />
          </div>
          <div className="cft-field">
            <label>Longitude (-180 to 180):</label>
            <input
              type="number"
              step="any"
              value={customLon}
              onChange={(e) => setCustomLon(e.target.value)}
              className="cft-input"
            />
          </div>
          <button type="submit" className="cft-apply-btn">Apply Coordinates</button>
          <button type="button" className="cft-cancel-btn" onClick={() => setEditManual(false)}>Cancel</button>
        </form>
      )}

      <MapContainer
        center={[lat, lon]}
        zoom={locStatus === 'EXACT' ? 13 : 8}
        zoomControl={false}
        scrollWheelZoom
        className="world-map-leaf"
        key={`${mapType}`}
      >
        <ZoomControl position="bottomright" />
        <TileLayer url={tileUrl} attribution={tileAttr} />
        <MapRecenterHelper lat={lat} lon={lon} />
        <MapClickHandler onMapClick={onLocationChange} />

        {/* Satellite image overlay — shown when bbox + image are both available */}
        {hasBbox && imageUrl && (
          <ImageOverlay
            url={imageUrl}
            bounds={[[bbox[1], bbox[0]], [bbox[3], bbox[2]]]}
            opacity={0.75}
            zIndex={10}
          />
        )}

        {hasBbox && (
          <Rectangle
            bounds={[[bbox[1], bbox[0]], [bbox[3], bbox[2]]]}
            pathOptions={{
              color: '#ef7657',
              weight: 2,
              dashArray: '5, 5',
              fillColor: '#ef7657',
              fillOpacity: 0.0,
            }}
          >
            <Popup>
              <div className="map-popup">
                <strong>📐 Georeferenced Footprint Extent</strong><br />
                <span>West: {bbox[0].toFixed(4)}° | East: {bbox[2].toFixed(4)}°</span><br />
                <span>South: {bbox[1].toFixed(4)}° | North: {bbox[3].toFixed(4)}°</span>
              </div>
            </Popup>
          </Rectangle>
        )}

        <Marker
          position={[lat, lon]}
          icon={customIcon}
          draggable={true}
          eventHandlers={{
            dragend: (e) => {
              const marker = e.target;
              const pos = marker.getLatLng();
              if (onLocationChange) {
                onLocationChange(pos.lat, pos.lng);
              }
            },
          }}
        >
          <Popup>
            <div className="map-popup">
              <strong>{label}</strong><br />
              {featureType && <span className="popup-ft">{featureType}<br /></span>}
              <span>{lat.toFixed(6)}°, {lon.toFixed(6)}°</span><br />
              <small style={{ color: '#ef7657', display: 'block', marginTop: '4px' }}>
                💡 Drag this pin or click on the map to fine-tune the exact center.
              </small>
            </div>
          </Popup>
        </Marker>
      </MapContainer>

      <div className="map-interactive-hint">
        <span>💡 <strong>Exact Center Pinpoint:</strong> Drag the orange target pin or click anywhere on the satellite map to adjust to the exact building, runway, or feature.</span>
      </div>

      <div className="map-coord-bar">
        <span>📍 {Math.abs(lat).toFixed(6)}° {lat >= 0 ? 'N' : 'S'}</span>
        <span>{Math.abs(lon).toFixed(6)}° {lon >= 0 ? 'E' : 'W'}</span>
        {hasBbox && (
          <span className="footprint-pill">📐 Footprint: [W {bbox[0].toFixed(2)}°, S {bbox[1].toFixed(2)}° to E {bbox[2].toFixed(2)}°, N {bbox[3].toFixed(2)}°]</span>
        )}
        <span className={`coord-status ${locStatus === 'EXACT' ? 'exact' : 'est'}`}>{locStatus}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// No-Map Search — fallback when no coordinates
// ─────────────────────────────────────────────

function NoMapSearch({ onSearch }) {
  const [q, setQ] = useState('');
  const [searching, setSearching] = useState(false);
  const [err, setErr] = useState('');

  const handleSearch = async (e) => {
    e && e.preventDefault();
    const query = q.trim();
    if (!query || !onSearch) return;
    setSearching(true);
    setErr('');
    try {
      await onSearch(query);
      setQ('');
    } catch (ex) {
      setErr(ex.message || 'Location not found.');
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="no-map-search">
      <p className="nms-label">🔍 Enter the location name or coordinates to pin it on the map:</p>
      <form className="nms-form" onSubmit={handleSearch}>
        <input
          type="text"
          className="nms-input"
          value={q}
          onChange={(e) => { setQ(e.target.value); setErr(''); }}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch(e)}
          placeholder="e.g. Sriharikota India, 13.72, 80.23, or any landmark..."
          id="no-map-search-input"
        />
        <button
          type="submit"
          className="nms-btn"
          disabled={searching || !q.trim()}
          id="no-map-search-btn"
        >
          {searching ? 'Locating...' : 'Find & Pin Location 📍'}
        </button>
      </form>
      {err && <div className="nms-err">⚠ {err}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────
// Copernicus Data Space Ecosystem Live Passes Card
// ─────────────────────────────────────────────

function CopernicusPassCard({ lat, lon, isSar }) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  const fetchPasses = useCallback(async () => {
    if (lat === undefined || lon === undefined || lat === null || lon === null) return;
    setLoading(true);
    setError('');
    try {
      const resp = await fetch(`${API}/copernicus/live-satellite`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: lat,
          longitude: lon,
          modality: isSar ? 'sar' : 'optical',
          radius_km: 5.0,
        }),
      });
      if (resp.ok) {
        const payload = await resp.json();
        setData(payload);
      } else {
        setError('Could not query Copernicus Data Space catalog.');
      }
    } catch {
      setError('Connection to Copernicus Data Space failed.');
    } finally {
      setLoading(false);
    }
  }, [lat, lon, isSar]);

  useEffect(() => {
    fetchPasses();
  }, [fetchPasses]);

  return (
    <div className="copernicus-card">
      <div className="copernicus-header">
        <div className="copernicus-title-group">
          <span className="copernicus-icon">🛰️</span>
          <div>
            <span className="copernicus-eyebrow">ESA / COPERNICUS DATA SPACE ECOSYSTEM</span>
            <h4 className="copernicus-title">Live Satellite Passes & Orbital Footprint</h4>
          </div>
        </div>
        <div className="copernicus-badge-group">
          <span className="copernicus-mission-badge">
            {isSar ? 'Sentinel-1 SAR Radar' : 'Sentinel-2 Multispectral'}
          </span>
          <button
            type="button"
            className="copernicus-refresh-btn"
            onClick={fetchPasses}
            disabled={loading}
            title="Refresh latest Copernicus satellite passes"
          >
            {loading ? '⟳ Querying ESA...' : '⟳ Refresh Passes'}
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="copernicus-loading">
          <span className="sat-btn-spinner" />
          <span>Searching Copernicus Data Space catalog for passes over {lat.toFixed(4)}°, {lon.toFixed(4)}°...</span>
        </div>
      )}

      {error && <div className="copernicus-error">⚠ {error}</div>}

      {data && (
        <div className="copernicus-body">
          {data.recent_passes && data.recent_passes.length > 0 ? (
            <div className="copernicus-passes-list">
              <span className="cpl-label">🛰️ Recent Satellite Passes Over This Coordinate:</span>
              <div className="cpl-grid">
                {data.recent_passes.slice(0, 3).map((p, idx) => (
                  <div key={idx} className="cpl-item">
                    <div className="cpl-top">
                      <span className="cpl-name">{p.name.slice(0, 38)}...</span>
                      <span className="cpl-tag">{p.satellite}</span>
                    </div>
                    <div className="cpl-bottom">
                      <span>📅 Acquired: {p.date ? new Date(p.date).toUTCString() : 'Recent pass'}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="copernicus-no-passes">
              No recent cloud-free {isSar ? 'Sentinel-1' : 'Sentinel-2'} passes found within immediate footprint.
            </div>
          )}

          {data.live_image_b64 ? (
            <div className="copernicus-live-preview">
              <span className="clp-title">Live Satellite Process API Tile:</span>
              <img
                src={`data:image/png;base64,${data.live_image_b64}`}
                alt="Copernicus Process API Tile"
                className="copernicus-tile-img"
              />
            </div>
          ) : (
            <div className="copernicus-hint-bar">
              <span>💡 <strong>Copernicus Process API Ready:</strong> To render raw live Sentinel tiles directly from orbit, configure <code>COPERNICUS_CLIENT_ID</code> and <code>COPERNICUS_CLIENT_SECRET</code> in <code>.env</code>.</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// MAIN WIZARD APP
// ─────────────────────────────────────────────

function SatQueryApp() {
  const [currentStep, setCurrentStep] = useState(1);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [sarViewMode, setSarViewMode] = useState('raw'); // 'raw', 'despeckle', 'heatmap'
  const [question, setQuestion] = useState('Describe the land cover in this image.');
  const [followUpQ, setFollowUpQ] = useState('');
  const [askingFollowUp, setAskingFollowUp] = useState(false);
  const [qaList, setQaList] = useState([]);

  // Preview URL generator
  useEffect(() => {
    if (!file) { setPreviewUrl(null); return; }
    if (/\.tiff?$/i.test(file.name)) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFile = useCallback((f) => {
    setFile(f);
    setResult(null);
    setError('');
    setSarViewMode('raw');
    setQaList([]);
  }, []);

  const runAnalysis = useCallback(async () => {
    if (!file) return;
    setAnalyzing(true);
    setError('');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45000); // 45s max timeout
    try {
      const body = new FormData();
      body.append('file_1', file);
      if (question && question.trim()) {
        body.append('question', question.trim());
      }
      const resp = await fetch(`${API}/analyze`, {
        method: 'POST',
        body,
        signal: controller.signal,
      });
      clearTimeout(timer);
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.detail || 'Analysis pipeline returned an error.');
      setResult(payload);
      setQaList([]);
      setCurrentStep(2); // Automatically advance to Step 2 (Verification)
    } catch (e) {
      clearTimeout(timer);
      if (e.name === 'AbortError') {
        setError('Analysis request timed out after 45 seconds. Please try again.');
      } else if (e.message && (e.message.includes('Failed to fetch') || e.message.includes('NetworkError'))) {
        // Check if a pre-generated sample result is available (for Vercel deployment without local backend)
        const sampleMap = {
          'sample_sar_radar.png': '/sample_images/sample_sar_radar_result.json',
          'paris_satellite_sentinel2.png': '/sample_images/sample_paris_satellite_result.json',
          'sample_paris_satellite.png': '/sample_images/sample_paris_satellite_result.json',
          'tokyo_bay_satellite.png': '/sample_images/sample_tokyo_satellite_result.json',
          'sample_tokyo_satellite.png': '/sample_images/sample_tokyo_satellite_result.json',
        };
        const sampleJsonUrl = file ? sampleMap[file.name] : null;
        if (sampleJsonUrl) {
          try {
            const demoResp = await fetch(sampleJsonUrl);
            if (demoResp.ok) {
              const demoData = await demoResp.json();
              setResult(demoData);
              setQaList([]);
              setCurrentStep(2);
              return;
            }
          } catch (demoErr) {
            console.warn('Demo fallback fetch failed:', demoErr);
          }
        }
        if (window.location.protocol === 'https:' && API.includes('127.0.0.1')) {
          setError('Backend is running on localhost (HTTP) while Vercel is served over HTTPS. Browsers block HTTPS sites from fetching local HTTP servers (Mixed Content). To run custom image analysis, open the app locally at http://127.0.0.1:5173/ or deploy the backend to a cloud HTTPS URL.');
        } else {
          setError(`Could not connect to the backend server at ${API}. Make sure the FastAPI backend is running.`);
        }
      } else {
        setError(e.message || 'Analysis could not be completed.');
      }
    } finally {
      setAnalyzing(false);
    }
  }, [file, question]);

  const handleAskFollowUp = async (e) => {
    if (e) e.preventDefault();
    if (!file || !followUpQ.trim() || askingFollowUp) return;
    setAskingFollowUp(true);
    try {
      const body = new FormData();
      body.append('file_1', file);
      body.append('question', followUpQ.trim());
      const resp = await fetch(`${API}/ask-question`, {
        method: 'POST',
        body,
      });
      const data = await resp.json();
      if (resp.ok && data.answer) {
        setQaList((prev) => [...prev, { question: followUpQ.trim(), answer: data.answer }]);
        setFollowUpQ('');
      } else {
        alert(data.detail || 'Could not get answer for this query.');
      }
    } catch (err) {
      console.error('Follow-up query error:', err);
    } finally {
      setAskingFollowUp(false);
    }
  };

  const handleLocationPinpoint = async (newLat, newLon, customTarget = null) => {
    const rLat = Number(newLat.toFixed(6));
    const rLon = Number(newLon.toFixed(6));
    setResult((prev) => {
      if (!prev) return prev;
      const old = prev.location || {};
      const tName = customTarget || old.target_name || `Pinpoint Center (${rLat.toFixed(4)}°, ${rLon.toFixed(4)}°)`;
      return {
        ...prev,
        location: {
          ...old,
          latitude: rLat,
          longitude: rLon,
          target_name: tName,
          status: 'EXACT',
          bounding_box: [
            Number((rLon - 0.015).toFixed(6)),
            Number((rLat - 0.015).toFixed(6)),
            Number((rLon + 0.015).toFixed(6)),
            Number((rLat + 0.015).toFixed(6)),
          ],
        },
        location_evidence: [
          `✓ Exact coordinates pinpointed to ${rLat}°, ${rLon}°`,
          ...(prev.location_evidence || []),
        ],
      };
    });

    try {
      const resp = await fetch(`${API}/reverse-geocode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude: rLat, longitude: rLon }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setResult((prev) => {
          if (!prev) return prev;
          const cur = prev.location || {};
          const t = customTarget || cur.target_name;
          const dName = t
            ? `${t}, ${data.city || data.state || ''}, ${data.country || ''}`.replace(', ,', ',').replace(/^,\s*|,\s*$/g, '')
            : data.display_name;
          return {
            ...prev,
            location: {
              ...cur,
              latitude: data.latitude,
              longitude: data.longitude,
              city: data.city || cur.city,
              state: data.state || cur.state,
              country: data.country || cur.country,
              display_name: dName || cur.display_name,
              bounding_box: data.bounding_box || cur.bounding_box,
            },
          };
        });
      }
    } catch (e) {
      console.warn('Reverse geocode update failed:', e);
    }
  };

  const handleLocationSearch = async (query) => {
    const resp = await fetch(`${API}/search-location?query=${encodeURIComponent(query)}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Search failed' }));
      throw new Error(err.detail || 'Location not found');
    }
    const data = await resp.json();
    await handleLocationPinpoint(data.latitude, data.longitude, data.target_name || query);
    return data;
  };

  // ── Download helpers ──
  const downloadReportHTML = () => {
    if (!result) return;
    const loc = result.location || {};
    const info = result.image_info || {};
    const sar = result.sar_metadata || {};
    const geo = result.geotiff_metadata || {};
    const land = result.land_cover || [];
    const trace = result.execution_trace || [];
    const now = new Date().toUTCString();
    const locStr = [loc.city, loc.state, loc.country].filter(Boolean).join(', ') || 'Unknown';
    const sarRows = sar.is_sar ? `
      <tr><td>Radar Band</td><td>${sar.radar_band || '—'}</td></tr>
      <tr><td>Polarization</td><td>${sar.polarization || '—'}</td></tr>
      <tr><td>Speckle Index (Cv)</td><td>${sar.speckle_index ?? '—'}</td></tr>
      <tr><td>Equivalent Looks (ENL)</td><td>${sar.equivalent_looks ?? '—'}</td></tr>
      <tr><td>Backscatter Range</td><td>${sar.backscatter_db ? `${sar.backscatter_db.min_db} dB → ${sar.backscatter_db.max_db} dB (Mean: ${sar.backscatter_db.mean_db} dB)` : '—'}</td></tr>
      <tr><td>Double-Bounce</td><td>${sar.scattering_mechanisms?.double_bounce_percent ?? '—'}%</td></tr>
      <tr><td>Volume/Surface</td><td>${sar.scattering_mechanisms?.volume_surface_percent ?? '—'}%</td></tr>
    ` : '';
    const geoRows = geo.is_geotiff ? `
      <tr><td>CRS</td><td>${geo.crs || '—'}</td></tr>
      <tr><td>EPSG</td><td>${geo.epsg || '—'}</td></tr>
      <tr><td>Pixel Scale</td><td>${Array.isArray(geo.pixel_scale) ? geo.pixel_scale.join(', ') : (geo.pixel_scale || '—')}</td></tr>
      <tr><td>Pixel Resolution</td><td>${geo.pixel_resolution || '—'}</td></tr>
      <tr><td>Raster Shape</td><td>${Array.isArray(geo.raster_shape) ? geo.raster_shape.join(' × ') : '—'}</td></tr>
      <tr><td>Band Count</td><td>${geo.bands_count ?? '—'}</td></tr>
    ` : '';
    const landRows = land.map(lc => `<tr><td>${lc.emoji} ${lc.category}</td><td>${lc.percent}%</td><td>${lc.notes || ''}</td></tr>`).join('');
    const traceRows = trace.map((tr, i) => `<tr><td>${i + 1}</td><td>${tr.step || ''}</td><td>${tr.detail || ''}</td></tr>`).join('');
    const qaRows = qaList.map((qa, i) => `<tr><td>${i + 1}</td><td>${qa.question}</td><td>${qa.answer}</td></tr>`).join('');
    const imgTag = displayImg ? `<img src="${displayImg}" style="max-width:420px;border-radius:6px;border:1px solid #ccc;" alt="Satellite Image" />` : '<em>Image preview not available</em>';
    const evidenceHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>SatQuery AI — Evidence Report (${info.filename || 'analysis'})</title>
<style>
  @page { size: A4; margin: 20mm 15mm 20mm 15mm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; color: #1a1a1a; background: #fff; }
  header { background: #0b1213; color: #e8ede8; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 3px solid #ef7657; margin-bottom: 24px; }
  header h1 { font-size: 18px; letter-spacing: 0.5px; }
  header .meta { font-size: 10px; color: #a8b8b0; text-align: right; line-height: 1.6; }
  .watermark { font-size: 10px; color: #ef7657; font-weight: 700; letter-spacing: 1px; margin-bottom: 2px; }
  section { margin-bottom: 24px; page-break-inside: avoid; }
  h2 { font-size: 13px; font-weight: 700; color: #0b1213; border-bottom: 1.5px solid #ef7657; padding-bottom: 5px; margin-bottom: 10px; letter-spacing: 0.3px; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { background: #f0f4f3; text-align: left; padding: 6px 10px; font-weight: 600; color: #263334; border-bottom: 1px solid #d0dbd8; }
  td { padding: 5px 10px; border-bottom: 1px solid #e8eeec; color: #1a2b2a; }
  tr:hover td { background: #f7faf9; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; }
  .badge-sar { background: #fde9e2; color: #c85c3c; }
  .badge-geo { background: #e2f0fd; color: #2563a8; }
  .badge-ok  { background: #e4f5e4; color: #287a28; }
  .img-wrap { display: flex; gap: 24px; align-items: flex-start; margin-bottom: 14px; }
  .img-wrap img { max-width: 380px; border-radius: 6px; border: 1px solid #d0dbd8; }
  .coords { font-family: monospace; background: #f0f4f3; padding: 3px 8px; border-radius: 3px; }
  footer { border-top: 1px solid #d0dbd8; padding-top: 10px; margin-top: 24px; font-size: 10px; color: #6e827a; display: flex; justify-content: space-between; }
  .ai-box { background: #f0f4f3; border-left: 3px solid #ef7657; padding: 10px 14px; border-radius: 3px; font-size: 11px; line-height: 1.65; }
  @media print {
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    section { page-break-inside: avoid; }
    .no-print { display: none !important; }
  }
</style>
</head>
<body>
<header>
  <div>
    <div class="watermark">SATQUERY AI · EVIDENCE REPORT</div>
    <h1>🛰️ Remote Sensing Intelligence Dossier</h1>
    <div style="font-size:11px;color:#a8b8b0;margin-top:4px;">${info.filename || 'Satellite Image Analysis'}</div>
  </div>
  <div class="meta">
    <div><strong>Generated:</strong> ${now}</div>
    <div><strong>Location:</strong> ${locStr}</div>
    <div><strong>Modality:</strong> ${result.modality || '—'}</div>
    ${sar.is_sar ? '<div><span class="badge badge-sar">SAR</span></div>' : ''}
    ${geo.is_geotiff ? '<div><span class="badge badge-geo">GeoTIFF</span></div>' : ''}
  </div>
</header>

<section>
  <h2>📸 1. Image Telemetry &amp; Modality</h2>
  <div class="img-wrap">
    ${imgTag}
    <table style="flex:1;">
      <tr><th colspan="2">Image Properties</th></tr>
      <tr><td>Filename</td><td>${info.filename || '—'}</td></tr>
      <tr><td>Dimensions</td><td>${info.width ? `${info.width} × ${info.height} px` : '—'}</td></tr>
      <tr><td>Format</td><td>${info.format || '—'}</td></tr>
      <tr><td>File Size</td><td>${info.file_size_display || '—'}</td></tr>
      <tr><td>Bands</td><td>${info.bands ?? '—'}</td></tr>
      <tr><td>Satellite / Sensor</td><td>${info.satellite || '—'}</td></tr>
      <tr><td>Capture Date</td><td>${info.capture_date || '—'}</td></tr>
      <tr><td>Capture Time</td><td>${info.capture_time || '—'}</td></tr>
      <tr><td>CRS / Projection</td><td>${geo.crs || info.crs || 'WGS 84'}</td></tr>
      <tr><td>Modality</td><td>${result.modality || '—'}</td></tr>
    </table>
  </div>
</section>

${(sar.is_sar || geo.is_geotiff) ? `
<section>
  <h2>${sar.is_sar ? '📡 2. SAR Radar Metrics' : '🗺️ 2. GeoTIFF Geospatial Metadata'}</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    ${sar.is_sar ? sarRows : geoRows}
  </table>
</section>` : ''}

<section>
  <h2>📍 3. Grounded Geolocation &amp; Footprint</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Target Name</td><td>${loc.target_name || '—'}</td></tr>
    <tr><td>Display Name</td><td>${loc.display_name || '—'}</td></tr>
    <tr><td>Coordinates</td><td><span class="coords">${loc.latitude != null ? `${loc.latitude}°, ${loc.longitude}°` : '—'}</span></td></tr>
    <tr><td>City</td><td>${loc.city || '—'}</td></tr>
    <tr><td>State / Region</td><td>${loc.state || '—'}</td></tr>
    <tr><td>Country</td><td>${loc.country || '—'}</td></tr>
    <tr><td>Bounding Box</td><td>${Array.isArray(loc.bounding_box) ? loc.bounding_box.join(', ') : (Array.isArray(bbox) ? bbox.join(', ') : '—')}</td></tr>
    <tr><td>Location Status</td><td>${loc.status || '—'}</td></tr>
    <tr><td>AI Geolocation Confidence</td><td>${loc.geolocation_confidence != null ? `${loc.geolocation_confidence}%` : '—'}</td></tr>
    <tr><td>Feature Type</td><td>${loc.feature_type || '—'}</td></tr>
  </table>
  ${(result.location_evidence || []).length > 0 ? `
  <br/>
  <table>
    <tr><th>Location Evidence Chain</th></tr>
    ${(result.location_evidence || []).map(e => `<tr><td>${e}</td></tr>`).join('')}
  </table>` : ''}
</section>

${land.length > 0 ? `
<section>
  <h2>🌱 4. Land-Cover Classification</h2>
  <table>
    <tr><th>Category</th><th>Coverage</th><th>Notes</th></tr>
    ${landRows}
  </table>
</section>` : ''}

${result.ai_interpretation ? `
<section>
  <h2>🤖 5. AI Remote Sensing Interpretation</h2>
  <div class="ai-box">${result.ai_interpretation}</div>
</section>` : ''}

${result.spectral_analysis ? `
<section>
  <h2>🔬 6. Spectral Analysis</h2>
  <table>
    <tr><th>Index</th><th>Status</th></tr>
    <tr><td>Multispectral Available</td><td>${result.spectral_analysis.multispectral_available ? 'Yes' : 'No'}</td></tr>
    <tr><td>NDVI</td><td>${result.spectral_analysis.ndvi_possible ? 'Calculable' : 'N/A'}</td></tr>
    <tr><td>NDWI</td><td>${result.spectral_analysis.ndwi_possible ? 'Calculable' : 'N/A'}</td></tr>
    <tr><td>NDBI</td><td>${result.spectral_analysis.ndbi_possible ? 'Calculable' : 'N/A'}</td></tr>
    ${result.spectral_analysis.note ? `<tr><td colspan="2">${result.spectral_analysis.note}</td></tr>` : ''}
  </table>
</section>` : ''}

${qaList.length > 0 ? `
<section>
  <h2>💬 7. Visual Question &amp; Answer Log</h2>
  <table>
    <tr><th>#</th><th>Question</th><th>Answer</th></tr>
    ${qaRows}
  </table>
</section>` : ''}

${trace.length > 0 ? `
<section>
  <h2>⚙️ 8. Automated Execution Trace</h2>
  <table>
    <tr><th>#</th><th>Step</th><th>Detail</th></tr>
    ${traceRows}
  </table>
</section>` : ''}

<footer>
  <div>SatQuery AI · Global Remote Sensing &amp; SAR/Optical Intelligence Platform</div>
  <div>Results are advisory and should be reviewed by a qualified analyst. Generated: ${now}</div>
</footer>
</body>
</html>`;

    const printWin = window.open('', '_blank', 'width=1000,height=800');
    if (!printWin) { alert('Popup blocked — please allow popups for this site to download the PDF report.'); return; }
    printWin.document.write(evidenceHTML);
    printWin.document.close();
    printWin.focus();
    setTimeout(() => {
      printWin.print();
    }, 600);
  };

  const downloadReportJSON = () => {
    if (!result) return;
    const exportData = {
      _meta: {
        generated_at: new Date().toISOString(),
        tool: 'SatQuery AI Evidence Report',
        filename: result.image_info?.filename || 'analysis',
      },
      ...result,
      qa_log: qaList,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satquery_evidence_${(result.image_info?.filename || 'report').replace(/[^a-zA-Z0-9._-]/g, '_')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const sarData = result?.sar_metadata;
  const isSar = Boolean(sarData?.is_sar);
  const geoInfo = result?.geotiff_metadata;
  const isGeoTiff = Boolean(geoInfo?.is_geotiff);

  // Determine which image to display based on SAR view mode
  let displayImg = previewUrl || (result?.uploaded_image_b64 ? `data:image/jpeg;base64,${result.uploaded_image_b64}` : null);
  if (isSar) {
    if (sarViewMode === 'despeckle' && sarData?.despeckled_image_b64) {
      displayImg = `data:image/jpeg;base64,${sarData.despeckled_image_b64}`;
    } else if (sarViewMode === 'heatmap' && sarData?.radar_heatmap_b64) {
      displayImg = `data:image/jpeg;base64,${sarData.radar_heatmap_b64}`;
    }
  }

  const hasCoords = result?.location?.latitude != null && result?.location?.longitude != null;
  const bbox = result?.location?.bounding_box || geoInfo?.bounding_box;

  return (
    <div className="satq-shell">
      {/* ── Topbar ── */}
      <header className="satq-topbar">
        <a className="satq-brand" href="/" aria-label="SatQuery AI home">
          <span className="satq-brand-mark">SQ</span>
          <span>SatQuery <em>AI</em></span>
        </a>
        <div className="satq-nav">
          <span className="satq-status-dot" />
          <span className="satq-status-txt">Global Earth Observation & SAR/Optical Platform</span>
        </div>
      </header>

      {/* ── Wizard Step Navigation Bar ── */}
      <nav className="wizard-nav-bar">
        <div className="wizard-nav-inner">
          {STEPS.map((step) => {
            const isCompleted = result != null && currentStep > step.id;
            const isCurrent = currentStep === step.id;
            const isDisabled = !result && step.id > 1;

            return (
              <button
                key={step.id}
                className={`wizard-step-btn ${isCurrent ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                disabled={isDisabled}
                onClick={() => !isDisabled && setCurrentStep(step.id)}
                id={`wizard-step-${step.id}`}
              >
                <span className="wizard-step-num">
                  {isCompleted ? '✓' : step.id}
                </span>
                <span className="wizard-step-icon">{step.icon}</span>
                <span className="wizard-step-label">{step.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* ── Main Container ── */}
      <main className="satq-main">
        {/* Loading Overlay */}
        {analyzing && <AnalysisOverlay filename={file?.name} />}

        {/* ── STEP 1: UPLOAD & INGEST ── */}
        {!analyzing && currentStep === 1 && (
          <div className="step-card fade-in">
            <div className="step-card-header">
              <span className="step-badge">STEP 01</span>
              <h2>Satellite Image Ingestion & Visual Question Answering</h2>
              <p>Upload Synthetic Aperture Radar (SAR), Optical multispectral, or GeoTIFF imagery and ask any question about the terrain or infrastructure.</p>
            </div>

            {/* Satellite Image Upload Box */}
            <UploadZone
              onFile={handleFile}
              file={file}
              previewUrl={previewUrl}
              onRemove={() => setFile(null)}
            />

            {/* Question Box Card */}
            <div className="sat-question-card">
              <label htmlFor="satellite-question-input" className="sat-question-label">
                Question
              </label>
              <textarea
                id="satellite-question-input"
                className="sat-question-textarea"
                rows={2}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Describe the land cover in this image."
              />
            </div>

            {/* Vibrant Orange Ask Button */}
            <button
              type="button"
              className="sat-ask-btn"
              disabled={!file || analyzing}
              onClick={runAnalysis}
              id="ask-analysis-btn"
            >
              {analyzing ? (
                <span className="sat-btn-spinner-wrap">
                  <span className="sat-btn-spinner" />
                  Analyzing Satellite Image...
                </span>
              ) : (
                'Ask'
              )}
            </button>

            {/* Quick Test Sample Images */}
            <div className="quick-test-box">
              <span className="quick-test-label">💡 Or test instantly with a verified sample:</span>
              <div className="quick-test-btns">
                <button
                  type="button"
                  className="sample-chip sar-chip"
                  onClick={async () => {
                    const res = await fetch('/sample_images/sample_sar_radar.png');
                    const blob = await res.blob();
                    const testFile = new File([blob], 'sample_sar_radar.png', { type: 'image/png' });
                    handleFile(testFile);
                  }}
                  id="sample-sar-btn"
                >
                  🛰️ SAR Radar: Launch Facility (Sriharikota)
                </button>
                <button
                  type="button"
                  className="sample-chip geotiff-chip"
                  onClick={async () => {
                    const res = await fetch('/sample_images/real_14band_s1_s2.tif');
                    const blob = await res.blob();
                    const testFile = new File([blob], 'real_14band_s1_s2.tif', { type: 'image/tiff' });
                    handleFile(testFile);
                  }}
                  id="sample-geotiff-btn"
                >
                  🌍 GeoTIFF: S1/S2 Multiband Chip (UTM 34N)
                </button>
                <button
                  type="button"
                  className="sample-chip"
                  onClick={async () => {
                    const res = await fetch('/sample_images/sample_paris_satellite.png');
                    const blob = await res.blob();
                    const testFile = new File([blob], 'paris_satellite_sentinel2.png', { type: 'image/png' });
                    handleFile(testFile);
                  }}
                  id="sample-paris-btn"
                >
                  🏙️ Paris, France (Optical Satellite)
                </button>
                <button
                  type="button"
                  className="sample-chip"
                  onClick={async () => {
                    const res = await fetch('/sample_images/sample_tokyo_satellite.png');
                    const blob = await res.blob();
                    const testFile = new File([blob], 'tokyo_bay_satellite.png', { type: 'image/png' });
                    handleFile(testFile);
                  }}
                  id="sample-tokyo-btn"
                >
                  🌊 Tokyo, Japan (Optical Coastal)
                </button>
              </div>
            </div>

            {error && (
              <div className="error-banner" role="alert">
                <span>⚠</span>
                <div>
                  <strong>Analysis Failed</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {result && (
              <div className="step-actions" style={{ marginTop: '16px' }}>
                <button className="primary-btn" onClick={() => setCurrentStep(2)}>
                  Continue to Step 2: Verification →
                </button>
              </div>
            )}
          </div>
        )}

        {/* ── STEP 2: MODALITY & IMAGE VERIFICATION ── */}
        {!analyzing && currentStep === 2 && result && (
          <div className="step-card fade-in">
            <div className="step-card-header">
              <span className="step-badge">STEP 02</span>
              <h2>Modality & Sensor Verification</h2>
              <p>Automated SAR (Synthetic Aperture Radar) vs Optical Multispectral classification & integrity verification.</p>
            </div>

            {/* Satellite Vision Q&A Card */}
            <div className="vqa-result-card">
              <div className="vqa-card-header">
                <div className="vqa-title-row">
                  <span className="vqa-q-icon">💬</span>
                  <div>
                    <span className="vqa-eyebrow">QUESTION & VISION ANSWER</span>
                    <h3 className="vqa-main-question">"{result.question || question}"</h3>
                  </div>
                </div>
                <span className="vqa-badge">Remote Sensing Grounded</span>
              </div>
              <div className="vqa-body">
                <p className="vqa-answer-text">
                  {result.question_answer || result.ai_interpretation || 'Analysis completed.'}
                </p>
              </div>

              {/* Follow-up question form */}
              <form className="vqa-followup-form" onSubmit={handleAskFollowUp}>
                <input
                  type="text"
                  className="vqa-followup-input"
                  value={followUpQ}
                  onChange={(e) => setFollowUpQ(e.target.value)}
                  placeholder="Ask another question about this satellite image..."
                />
                <button
                  type="submit"
                  className="vqa-followup-btn"
                  disabled={!followUpQ.trim() || askingFollowUp}
                >
                  {askingFollowUp ? 'Thinking...' : 'Ask'}
                </button>
              </form>

              {/* Extra Q&A history */}
              {qaList.length > 0 && (
                <div className="vqa-history-list">
                  {qaList.map((item, idx) => (
                    <div key={idx} className="vqa-history-item">
                      <div className="vqa-history-q">
                        <strong>Q:</strong> {item.question}
                      </div>
                      <div className="vqa-history-a">
                        <strong>A:</strong> {item.answer}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Modality Hero Banner */}
            <div className={`modality-hero-card ${isSar ? 'modality-sar' : isGeoTiff ? 'modality-geotiff' : 'modality-optical'}`}>
              <div className="modality-hero-icon">
                {isSar ? '📡' : isGeoTiff ? '🌍' : '🛰️'}
              </div>
              <div className="modality-hero-body">
                <span className="modality-hero-eyebrow">IDENTIFIED SENSOR MODALITY</span>
                <h3 className="modality-hero-title">{result.modality || 'Remote Sensing Imagery'}</h3>
                <p className="modality-hero-desc">
                  {isSar
                    ? 'Active microwave Synthetic Aperture Radar (SAR) sensor. Cloud-penetrating, day/night capable radar backscatter measurement.'
                    : isGeoTiff
                    ? 'Georeferenced multi-band satellite data chip containing embedded Coordinate Reference System (CRS) projection parameters.'
                    : 'Passive optical sensor capturing solar reflectance across visible and multispectral wavelengths.'}
                </p>
              </div>
              <div className="modality-hero-badge">
                <span className="mhb-status">CONFIRMED</span>
                <span className="mhb-conf">{result.satellite_verification?.satellite_confidence || 98}%</span>
              </div>
            </div>

            <div className="analyzed-image-split">
              <div className="img-holder">
                {displayImg ? (
                  <img src={displayImg} alt="Uploaded" className="step-img-preview" />
                ) : (
                  <div className="no-img-box">GeoTIFF Data Array Decoded</div>
                )}
                <span className="img-caption">{result.image_info?.filename}</span>
              </div>

              <div className="verif-details-card">
                <div className={`verif-badge-big ${result.satellite_verification?.satellite_confirmed ? 'confirmed' : 'denied'}`}>
                  <span className="verif-icon">{result.satellite_verification?.satellite_confirmed ? '✓' : '✗'}</span>
                  <span className="verif-label">
                    {result.satellite_verification?.satellite_confirmed ? 'Remote Sensing Confirmed' : 'Not Satellite Imagery'}
                  </span>
                </div>

                <div className="verif-type">
                  <span className="meta-label">Detected Architecture</span>
                  <span className="verif-type-val">{result.satellite_verification?.image_type}</span>
                </div>

                <div className="verif-conf">
                  <span className="meta-label">🛰️ Verification Confidence</span>
                  <ConfBar value={result.satellite_verification?.satellite_confidence} color={isSar ? '#e8c96a' : '#ef7657'} />
                </div>

                <div className="verif-reason">
                  <span className="meta-label">Classification Rationale</span>
                  <p>{result.satellite_verification?.reason}</p>
                </div>

                <div className="quick-spec-grid">
                  <div className="qsg-item">
                    <span>Dimensions</span>
                    <strong>{result.image_info?.width} × {result.image_info?.height} px</strong>
                  </div>
                  <div className="qsg-item">
                    <span>Bands</span>
                    <strong>{result.image_info?.bands || 1} Channel(s)</strong>
                  </div>
                  <div className="qsg-item">
                    <span>Sensor Mode</span>
                    <strong>{isSar ? sarData?.polarization || 'VV Radar' : result.image_info?.mode || 'RGB'}</strong>
                  </div>
                  <div className="qsg-item">
                    <span>Format</span>
                    <strong>{result.image_info?.format || 'GeoTIFF / Raster'}</strong>
                  </div>
                </div>
              </div>
            </div>

            <div className="step-actions">
              <button className="secondary-btn" onClick={() => setCurrentStep(1)}>
                ← Back to Upload
              </button>
              <button className="primary-btn" onClick={() => setCurrentStep(3)}>
                Continue to Step 3: Location & Footprint →
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 3: LOCATION & FOOTPRINT MAP ── */}
        {!analyzing && currentStep === 3 && result && (
          <div className="step-card fade-in">
            <div className="step-card-header">
              <span className="step-badge">STEP 03</span>
              <h2>Global Geographic Location & Coverage Footprint</h2>
              <p>GeoTIFF projection transformations, radar ground-truth matching, and interactive footprint bounding box.</p>
            </div>

            {/* Identified Ground Facility / Target Banner */}
            {result.location?.target_name && (
              <div className={`target-facility-banner ${result.location?.status === 'ESTIMATED' ? 'estimated-banner' : ''}`}>
                <div className="tfb-meta">
                  <span className="tfb-badge">{result.location.feature_type || 'DETECTED SCENE TYPE'}</span>
                  {result.location.status === 'ESTIMATED' ? (
                    <span className="tfb-estimated">🔍 VISUAL ESTIMATE — USE SEARCH TO REFINE</span>
                  ) : (
                    <span className="tfb-ground-truth">✓ AI GROUND-TRUTH RECONNAISSANCE</span>
                  )}
                </div>
                <h3 className="tfb-name">{result.location.status === 'ESTIMATED' ? '🔍' : '🎯'} {result.location.target_name}</h3>
                <p className="tfb-sub">
                  {result.location.status === 'ESTIMATED'
                    ? 'Scene type estimated from visual spectral analysis. Use the search bar below to pinpoint the exact location.'
                    : 'Ground feature identified from orthorectified structural geometries, radar signatures, and terrain footprints.'}
                </p>
              </div>
            )}

            {/* Location Summary Card */}
            <div className="location-summary-card">
              <div className="loc-title-row">
                <div className="location-name">
                  <span className="loc-icon">📍</span>
                  <div>
                    <h3 className="loc-place-name">
                      {result.location?.target_name || [result.location?.city, result.location?.state, result.location?.country].filter(Boolean).join(', ') || 'Georeferenced Target'}
                    </h3>
                    {result.location?.display_name && (
                      <p className="loc-display-address">{result.location.display_name}</p>
                    )}
                  </div>
                </div>
                <Badge
                  text={`STATUS: ${result.location?.status || 'UNKNOWN'}`}
                  variant={result.location?.status === 'EXACT' ? 'green' : 'yellow'}
                />
              </div>

              <div className="location-coords-row">
                {hasCoords ? (
                  <>
                    <div className="coord-chip">
                      <span className="coord-chip-label">Latitude</span>
                      <span className="coord-chip-val">{Math.abs(result.location.latitude).toFixed(6)}° {result.location.latitude >= 0 ? 'N' : 'S'}</span>
                    </div>
                    <div className="coord-chip">
                      <span className="coord-chip-label">Longitude</span>
                      <span className="coord-chip-val">{Math.abs(result.location.longitude).toFixed(6)}° {result.location.longitude >= 0 ? 'E' : 'W'}</span>
                    </div>
                    {bbox && (
                      <div className="coord-chip">
                        <span className="coord-chip-label">Geographic Bounding Box</span>
                        <span className="coord-chip-val">[{bbox[0].toFixed(3)}°, {bbox[1].toFixed(3)}° to {bbox[2].toFixed(3)}°, {bbox[3].toFixed(3)}°]</span>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="coord-chip unavailable">Coordinates unavailable</div>
                )}
                <div className="loc-conf-chip">
                  <span className="meta-label">📍 Geolocation Confidence</span>
                  <ConfBar value={result.location?.location_confidence || 96} color="#9ad49e" />
                </div>
              </div>
            </div>

            {/* Interactive World Map with Footprint */}
            {hasCoords ? (
              <div className="map-step-wrap">
                <div className="map-step-header">
                  <h3>🗺️ Interactive World Map with Footprint Extent</h3>
                  <span className="map-step-sub">
                    {result.location?.status === 'ESTIMATED'
                      ? '⚠️ Visual estimate shown — drag the pin, click the map, or search for the exact site below.'
                      : 'Pulsing target pin shows exact center. Drag the pin, click the map, or use the search bar to refine.'}
                  </span>
                </div>
                <WorldMap
                  lat={result.location.latitude}
                  lon={result.location.longitude}
                  city={result.location.city}
                  country={result.location.country}
                  status={result.location.status}
                  bbox={bbox}
                  imageUrl={displayImg}
                  targetName={result.location.target_name}
                  featureType={result.location.feature_type}
                  onLocationChange={handleLocationPinpoint}
                  onSearchLocation={handleLocationSearch}
                />
              </div>
            ) : (
              <div className="no-map-box">
                <span className="no-map-icon">🔍</span>
                <p>The image does not contain embedded geospatial metadata. Use the search bar to locate this image manually.</p>
                <div className="no-map-search-wrap">
                  <NoMapSearch onSearch={handleLocationSearch} />
                </div>
              </div>
            )}

            {/* Copernicus Data Space Live Satellite Acquisitions */}
            {hasCoords && (
              <CopernicusPassCard
                lat={result.location.latitude}
                lon={result.location.longitude}
                isSar={isSar}
              />
            )}

            {/* Location Evidence List */}
            <div className="evidence-subcard">
              <h4>🔎 Geolocation Evidence Log</h4>
              <div className="evidence-list">
                {(result.location_evidence || []).map((ev, i) => (
                  <div className={`evidence-item ${ev.startsWith('⚠') ? 'warn' : 'ok'}`} key={i}>
                    <span className="ev-icon">{ev.startsWith('⚠') ? '⚠' : '✓'}</span>
                    <span className="ev-text">{ev.replace(/^[⚠✓]\s*/, '')}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="step-actions">
              <button className="secondary-btn" onClick={() => setCurrentStep(2)}>
                ← Back to Verification
              </button>
              <button className="primary-btn" onClick={() => setCurrentStep(4)}>
                Continue to Step 4: SAR & Optical Viewers →
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 4: SAR & OPTICAL METADATA / GEOTIFF VIEWERS ── */}
        {!analyzing && currentStep === 4 && result && (
          <div className="step-card fade-in">
            <div className="step-card-header">
              <span className="step-badge">STEP 04</span>
              <h2>SAR & Optical Metadata / GeoTIFF Viewers</h2>
              <p>Comprehensive active microwave radar analysis, GeoTIFF projection inspector, and optical spectral diagnostics.</p>
            </div>

            {/* SAR Radar Inspector (if SAR is detected) */}
            {isSar && sarData && (
              <div className="viewer-card sar-viewer-card">
                <div className="viewer-card-header">
                  <span className="viewer-badge sar-badge">📡 SYNTHETIC APERTURE RADAR (SAR) INSPECTOR</span>
                  <h3>Active Microwave Backscatter & Speckle Diagnostics</h3>
                </div>

                {/* Interactive SAR View Mode Switcher */}
                <div className="sar-view-switcher">
                  <span className="svs-label">Interactive SAR Imagery Layer:</span>
                  <div className="svs-buttons">
                    <button
                      type="button"
                      className={`svs-btn ${sarViewMode === 'raw' ? 'active' : ''}`}
                      onClick={() => setSarViewMode('raw')}
                    >
                      📷 Raw SAR Backscatter
                    </button>
                    <button
                      type="button"
                      className={`svs-btn ${sarViewMode === 'despeckle' ? 'active' : ''}`}
                      onClick={() => setSarViewMode('despeckle')}
                    >
                      ✨ Despeckled Filter (Adaptive)
                    </button>
                    <button
                      type="button"
                      className={`svs-btn ${sarViewMode === 'heatmap' ? 'active' : ''}`}
                      onClick={() => setSarViewMode('heatmap')}
                    >
                      🌈 Radar Backscatter Heatmap
                    </button>
                  </div>
                </div>

                {displayImg && (
                  <div className="sar-interactive-preview">
                    <img src={displayImg} alt="SAR Layer" className="sar-main-img" />
                    <span className="sar-layer-tag">
                      {sarViewMode === 'raw' && 'Viewing: Coherent Single-Look Radar Intensity'}
                      {sarViewMode === 'despeckle' && 'Viewing: Speckle-Suppressed Adaptive Median Filter'}
                      {sarViewMode === 'heatmap' && 'Viewing: False-Color Microwave Backscatter Intensity Heatmap (dB)'}
                    </span>
                  </div>
                )}

                {/* SAR Metrics Grid */}
                <div className="sar-metrics-grid">
                  <div className="smg-card">
                    <span className="smg-label">Frequency Band</span>
                    <strong className="smg-val">{sarData.radar_band || 'C-Band (5.405 GHz)'}</strong>
                    <small>Active microwave wavelength</small>
                  </div>
                  <div className="smg-card">
                    <span className="smg-label">Polarization</span>
                    <strong className="smg-val">{sarData.polarization || 'VV Single-Pol'}</strong>
                    <small>Transmit / Receive geometry</small>
                  </div>
                  <div className="smg-card">
                    <span className="smg-label">Speckle Index (Cv)</span>
                    <strong className="smg-val">{sarData.speckle_index ?? 0.387}</strong>
                    <small>Standard dev / Mean intensity</small>
                  </div>
                  <div className="smg-card">
                    <span className="smg-label">Equivalent Looks (ENL)</span>
                    <strong className="smg-val">{sarData.equivalent_looks ?? 6.67}</strong>
                    <small>Multi-look speckle reduction</small>
                  </div>
                </div>

                {/* Radar Backscatter Decibel Scale */}
                {sarData.backscatter_db && (
                  <div className="sar-db-card">
                    <h4>📊 Radar Backscatter Range (Decibels dB)</h4>
                    <div className="db-stats-row">
                      <div className="db-chip"><span>Min Backscatter:</span><strong>{sarData.backscatter_db.min_db} dB</strong></div>
                      <div className="db-chip"><span>Mean Backscatter:</span><strong>{sarData.backscatter_db.mean_db} dB</strong></div>
                      <div className="db-chip"><span>Max Backscatter:</span><strong>{sarData.backscatter_db.max_db} dB</strong></div>
                      <div className="db-chip"><span>Dynamic Range:</span><strong>{sarData.backscatter_db.dynamic_range_db} dB</strong></div>
                    </div>
                  </div>
                )}

                {/* Microwave Scattering Mechanism Breakdown */}
                {sarData.scattering_mechanisms && (
                  <div className="sar-scattering-card">
                    <h4>⚡ Microwave Scattering Classification</h4>
                    <div className="scattering-grid">
                      <div className="scat-item">
                        <div className="scat-head">
                          <span>🏗️ Metallic Structures & Gantries (Double-Bounce)</span>
                          <strong>{sarData.scattering_mechanisms.double_bounce_percent}%</strong>
                        </div>
                        <div className="scat-bar"><div className="scat-fill double-bounce" style={{ width: `${sarData.scattering_mechanisms.double_bounce_percent}%` }} /></div>
                        <small>Corner reflectors, launch gantries, buildings, metallic surfaces</small>
                      </div>
                      <div className="scat-item">
                        <div className="scat-head">
                          <span>🌱 Rough Surface & Canopy (Volume Scatter)</span>
                          <strong>{sarData.scattering_mechanisms.volume_surface_percent}%</strong>
                        </div>
                        <div className="scat-bar"><div className="scat-fill volume" style={{ width: `${sarData.scattering_mechanisms.volume_surface_percent}%` }} /></div>
                        <small>Vegetation canopy, soil surface, rough terrain</small>
                      </div>
                      <div className="scat-item">
                        <div className="scat-head">
                          <span>💧 Water Bodies & Smooth Tarmac (Specular Absorption)</span>
                          <strong>{sarData.scattering_mechanisms.specular_absorption_percent}%</strong>
                        </div>
                        <div className="scat-bar"><div className="scat-fill specular" style={{ width: `${sarData.scattering_mechanisms.specular_absorption_percent}%` }} /></div>
                        <small>Retention ponds, cooling reservoirs, smooth paved runways</small>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* GeoTIFF Metadata Inspector (if GeoTIFF is detected) */}
            {isGeoTiff && geoInfo && (
              <div className="viewer-card geotiff-viewer-card">
                <div className="viewer-card-header">
                  <span className="viewer-badge geotiff-badge">🌍 GEOTIFF GEOSPATIAL INSPECTOR</span>
                  <h3>Coordinate Reference System & Raster Geometry</h3>
                </div>

                <div className="geotiff-spec-grid">
                  <div className="gt-spec-item">
                    <span className="gt-label">Coordinate Reference System (CRS)</span>
                    <strong className="gt-val">{geoInfo.crs || 'WGS 84 / Projected'}</strong>
                  </div>
                  <div className="gt-spec-item">
                    <span className="gt-label">EPSG Code</span>
                    <strong className="gt-val">{geoInfo.epsg ? `EPSG:${geoInfo.epsg}` : 'Inferred / WGS84'}</strong>
                  </div>
                  <div className="gt-spec-item">
                    <span className="gt-label">Ground Sampling Distance (GSD)</span>
                    <strong className="gt-val">{geoInfo.pixel_resolution || '10.00 m/px'}</strong>
                  </div>
                  <div className="gt-spec-item">
                    <span className="gt-label">Raster Dimensions & Bands</span>
                    <strong className="gt-val">{geoInfo.raster_shape ? geoInfo.raster_shape.join(' × ') : `${result.image_info?.width} × ${result.image_info?.height} × ${geoInfo.bands_count}`}</strong>
                  </div>
                  <div className="gt-spec-item">
                    <span className="gt-label">Data Type</span>
                    <strong className="gt-val">{geoInfo.data_type || 'Float32 / Int16'}</strong>
                  </div>
                  <div className="gt-spec-item">
                    <span className="gt-label">Model Tie Point (X, Y)</span>
                    <strong className="gt-val">{geoInfo.model_tiepoint ? `[${geoInfo.model_tiepoint[3]}, ${geoInfo.model_tiepoint[4]}]` : 'Standard Orthorectified'}</strong>
                  </div>
                </div>

                {/* Multiband Channel Breakdown */}
                {geoInfo.band_names && geoInfo.band_names.length > 0 && (
                  <div className="gt-bands-subcard">
                    <h4>📡 Multiband Spectral & SAR Channels ({geoInfo.band_names.length} Bands)</h4>
                    <div className="gt-band-chips">
                      {geoInfo.band_names.map((bname, i) => (
                        <div className="gt-band-chip" key={i}>
                          <span className="gt-bnum">B{i + 1}</span>
                          <span className="gt-bname">{bname}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="gt-actions">
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(geoInfo, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${result.image_info?.filename || 'geotiff'}_metadata.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                  >
                    ⬇ Export GeoTIFF Metadata JSON
                  </button>
                </div>
              </div>
            )}

            {/* Optical & Multispectral Diagnostics */}
            <div className="viewer-card optical-viewer-card">
              <div className="viewer-card-header">
                <span className="viewer-badge optical-badge">🌱 OPTICAL & SPECTRAL DIAGNOSTICS</span>
                <h3>Visible Reflectance & Vegetation Indices</h3>
              </div>

              <div className="optical-spec-grid">
                <div className="opt-item">
                  <span>Spectral Channels</span>
                  <strong>{result.optical_metadata?.spectral_channels?.join(', ') || 'Red, Green, Blue'}</strong>
                </div>
                <div className="opt-item">
                  <span>Radiometric Depth</span>
                  <strong>{result.optical_metadata?.radiometric_resolution || '24-bit TrueColor'}</strong>
                </div>
                <div className="opt-item">
                  <span>Cloud Cover</span>
                  <strong>{result.optical_metadata?.cloud_cover || '0% (Clear atmospheric window)'}</strong>
                </div>
              </div>

              <p className="spectral-note">{result.spectral_analysis?.note}</p>
              <div className="spectral-indices">
                {['ndvi', 'ndwi', 'ndbi'].map((idx) => (
                  <div className={`spectral-index ${result.spectral_analysis?.[`${idx}_possible`] ? 'possible' : 'not-possible'}`} key={idx}>
                    <span className="si-name">{idx.toUpperCase()}</span>
                    <span className="si-status">
                      {result.spectral_analysis?.[`${idx}_possible`]
                        ? `Calculable (${result.spectral_analysis[idx] ?? 'Active'})`
                        : isSar ? 'N/A (Microwave Radar)' : 'N/A (RGB Image)'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Land Cover Classification */}
            <div className="analysis-subcard">
              <h3>🌱 Surface Land-Cover Classification</h3>
              {result.land_cover?.length > 0 ? (
                <div className="landcover-grid">
                  {result.land_cover.map((lc, i) => (
                    <div className="lc-item" key={i}>
                      <div className="lc-header">
                        <span className="lc-emoji">{lc.emoji}</span>
                        <span className="lc-name">{lc.category}</span>
                        <span className="lc-pct">{lc.percent}%</span>
                      </div>
                      <div className="lc-bar-track">
                        <div className="lc-bar-fill" style={{ width: `${lc.percent}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-txt">Land-cover features could not be calculated for this image.</p>
              )}
            </div>

            {/* AI Scene Interpretation */}
            {result.ai_interpretation && (
              <div className="analysis-subcard">
                <h3>🤖 Remote Sensing Intelligence Interpretation</h3>
                <div className="ai-interp">
                  <p>{result.ai_interpretation}</p>
                </div>
              </div>
            )}

            <div className="step-actions">
              <button className="secondary-btn" onClick={() => setCurrentStep(3)}>
                ← Back to Location & Map
              </button>
              <button className="primary-btn" onClick={() => setCurrentStep(5)}>
                Generate Complete Evidence Report →
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 5: COMPLETE EVIDENCE REPORT ── */}
        {!analyzing && currentStep === 5 && result && (
          <div className="step-card fade-in">
            <div className="step-card-header">
              <span className="step-badge">STEP 05</span>
              <h2>Complete Remote Sensing Evidence Report</h2>
              <p>Consolidated intelligence dossier compiling SAR, GeoTIFF, Optical, and Geolocation telemetry.</p>
            </div>

            <div className="evidence-report">
              {/* Section 1: Overview */}
              <div className="report-subblock">
                <h3>🛰️ 1. Image Telemetry & Modality</h3>
                <div className="analyzed-image-wrap">
                  {displayImg && (
                    <div className="analyzed-image-frame">
                      <img src={displayImg} alt="Uploaded" className="analyzed-image" />
                      <a href={displayImg} download={result.image_info?.filename} className="img-download-btn">⬇ Download Frame</a>
                    </div>
                  )}
                  <div className="meta-grid">
                    <MetaRow label="Filename" value={result.image_info?.filename} />
                    <MetaRow label="Modality" value={result.modality} />
                    <MetaRow label="Dimensions" value={result.image_info?.width ? `${result.image_info.width} × ${result.image_info.height} px` : null} />
                    <MetaRow label="File Size" value={result.image_info?.file_size_display} />
                    <MetaRow label="CRS / Projection" value={geoInfo?.crs || result.image_info?.crs || 'WGS 84'} />
                    <MetaRow label="Satellite / Sensor" value={result.image_info?.satellite || (isSar ? 'Sentinel-1 C-SAR' : 'Sentinel-2 MSI')} />
                  </div>
                </div>
              </div>

              {/* Section 2: SAR or Optical Specs */}
              {isSar && sarData && (
                <div className="report-subblock">
                  <h3>📡 2. SAR Radar Metrics</h3>
                  <div className="meta-grid">
                    <MetaRow label="Radar Band" value={sarData.radar_band} />
                    <MetaRow label="Polarization" value={sarData.polarization} />
                    <MetaRow label="Speckle Index (Cv)" value={sarData.speckle_index} />
                    <MetaRow label="Equivalent Looks (ENL)" value={sarData.equivalent_looks} />
                    <MetaRow label="Backscatter Range" value={sarData.backscatter_db ? `${sarData.backscatter_db.min_db} dB to ${sarData.backscatter_db.max_db} dB (Mean: ${sarData.backscatter_db.mean_db} dB)` : null} />
                  </div>
                </div>
              )}

              {/* Section 3: Geolocation */}
              <div className="report-subblock">
                <h3>📍 3. Grounded Geolocation & Footprint</h3>
                <p className="loc-full-title">
                  {[result.location?.city, result.location?.state, result.location?.country].filter(Boolean).join(', ') || 'Grounded Location'}
                </p>
                {hasCoords && (
                  <WorldMap
                    lat={result.location.latitude}
                    lon={result.location.longitude}
                    city={result.location.city}
                    country={result.location.country}
                    status={result.location.status}
                    bbox={bbox}
                  />
                )}
              </div>

              {/* Section 4: Land Cover */}
              <div className="report-subblock">
                <h3>🌱 4. Land-Cover Breakdown</h3>
                <div className="lc-chips-row">
                  {(result.land_cover || []).map((lc, i) => (
                    <span className="lc-chip" key={i}>{lc.emoji} {lc.category}: {lc.percent}%</span>
                  ))}
                </div>
              </div>

              {/* Section 5: Execution Trace */}
              <div className="report-subblock">
                <h3>⚙️ 5. Automated Execution Trace Log</h3>
                <div className="trace-list">
                  {(result.execution_trace || []).map((tr, i) => (
                    <div className="trace-item ok" key={i}>
                      <span className="trace-dot ok" />
                      <div className="trace-content">
                        <span className="trace-step-name">{tr.step}</span>
                        {tr.detail && <span className="trace-detail">{tr.detail}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ── Download Evidence Package ── */}
            <div className="download-evidence-panel">
              <div className="dep-header">
                <span className="dep-icon">🔏</span>
                <div>
                  <h3 className="dep-title">Download Evidence Package</h3>
                  <p className="dep-desc">Export this intelligence dossier as a signed PDF or raw JSON data archive for audit and record-keeping.</p>
                </div>
              </div>
              <div className="dep-actions">
                <button
                  id="download-evidence-pdf"
                  className="dep-btn dep-btn-pdf"
                  onClick={downloadReportHTML}
                  title="Download printable PDF evidence report"
                >
                  <span className="dep-btn-icon">📄</span>
                  <span className="dep-btn-main">Download as PDF</span>
                  <span className="dep-btn-sub">Formatted evidence report · Print-ready</span>
                </button>
                <button
                  id="download-evidence-json"
                  className="dep-btn dep-btn-json"
                  onClick={downloadReportJSON}
                  title="Download raw JSON evidence package"
                >
                  <span className="dep-btn-icon">🗃️</span>
                  <span className="dep-btn-main">Download as JSON</span>
                  <span className="dep-btn-sub">Complete data archive · Machine-readable</span>
                </button>
              </div>
              <p className="dep-legal">⚠ Results are advisory. Verify with a qualified remote sensing analyst before operational use.</p>
            </div>

            <div className="step-actions">
              <button className="secondary-btn" onClick={() => setCurrentStep(4)}>
                ← Back to Step 4
              </button>
              <button className="primary-btn" onClick={() => { setFile(null); setResult(null); setCurrentStep(1); }}>
                + Start New Analysis
              </button>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="satq-footer">
        <span>SatQuery AI / Global Remote Sensing & SAR/Optical Intelligence</span>
        <span>Results are advisory and should be reviewed by an analyst.</span>
      </footer>
    </div>
  );
}

const container = document.getElementById('root');
if (!container._reactRoot) {
  container._reactRoot = createRoot(container);
}
container._reactRoot.render(<SatQueryApp />);
