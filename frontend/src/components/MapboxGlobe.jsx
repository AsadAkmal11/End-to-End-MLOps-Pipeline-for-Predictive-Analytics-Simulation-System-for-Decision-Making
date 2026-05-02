import { useEffect, useRef } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

// ─── Token ───────────────────────────────────────────────────────────────────
const MAPBOX_TOKEN =
  "pk.eyJ1IjoiemF3YXJmYWhpbSIsImEiOiJjbW9vdDlpdmExaTByMm9zMWNjaHFiZGx1In0.LAJunS_5s8lapn1RTiiVPA";

// ─── Climate helpers ──────────────────────────────────────────────────────────
const REGION_COLORS = {
  arid: "#f59e0b",
  temperate: "#22d3ee",
  tropical: "#4ade80",
};

function getRainRegion(rain) {
  if (rain < 250) return "arid";
  if (rain <= 800) return "temperate";
  return "tropical";
}

// ─── Build GeoJSON FeatureCollection ─────────────────────────────────────────
function buildGeoJSON(cities) {
  return {
    type: "FeatureCollection",
    features: cities.map((city) => ({
      type: "Feature",
      // All city properties embedded — retrieved on click via e.features[0].properties
      properties: {
        name: city.name,
        temp: city.temp,
        rain: city.rain,
        humidity: city.humidity ?? 60,
        N: city.N ?? 75,
        P: city.P ?? 40,
        K: city.K ?? 38,
        lat: city.lat,
        lng: city.lng,
        region_type: getRainRegion(city.rain),
      },
      geometry: {
        type: "Point",
        coordinates: [city.lng, city.lat],
      },
    })),
  };
}

// ─── Popup HTML ───────────────────────────────────────────────────────────────
function buildPopupHtml(props) {
  const region = props.region_type || "temperate";
  const color = REGION_COLORS[region] || "#94a3b8";
  return `
    <div style="font-family:Inter,sans-serif;font-size:13px;color:#1e293b;min-width:150px">
      <strong style="font-size:14px;display:block;margin-bottom:5px">${props.name}</strong>
      <span style="
        background:${color}22;border:1px solid ${color}88;color:${color};
        border-radius:999px;padding:1px 9px;font-size:10px;font-weight:700;
        text-transform:uppercase;letter-spacing:.5px;display:inline-block;margin-bottom:6px
      ">${region}</span>
      <div style="color:#475569;font-size:11.5px;line-height:1.8">
        🌡️ <b>${props.temp}°C</b> &nbsp;|&nbsp; 🌧️ <b>${props.rain} mm</b><br/>
        💧 <b>${props.humidity}%</b> humidity
      </div>
      <div style="margin-top:6px;font-size:10px;color:#94a3b8;font-style:italic">
        Click to run ML crop analysis ↗
      </div>
    </div>`;
}

// ─── Globe auto-rotation ──────────────────────────────────────────────────────
function startAutoRotate(map) {
  let animating = true;
  let lastTime = 0;

  function frame(time) {
    if (!animating) return;
    const delta = time - lastTime;
    lastTime = time;
    // Skip large deltas (tab switch, etc.)
    if (delta > 0 && delta < 200) {
      map.rotateTo(map.getBearing() - delta * 0.004, { duration: 0 });
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  // Stop on any user interaction
  const stop = () => { animating = false; };
  map.once("mousedown", stop);
  map.once("touchstart", stop);
  map.once("wheel", stop);

  return () => { animating = false; };
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function MapboxGlobe({ cities, onCitySelect }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const popupRef = useRef(null);
  const stopRotateRef = useRef(null);
  const hoveredIdRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: "mapbox://styles/mapbox/satellite-streets-v12",
      projection: "globe",
      zoom: 1.5,
      center: [0, 20],
      antialias: true,
      // Prevent accidental map drag during marker click
      dragRotate: true,
    });

    mapRef.current = map;

    // Persistent popup (reused for every click)
    popupRef.current = new mapboxgl.Popup({
      closeButton: true,
      closeOnClick: false,
      className: "globe-popup",
      maxWidth: "240px",
      offset: 14,
    });

    // Controls
    map.addControl(new mapboxgl.NavigationControl({ showCompass: true }), "top-right");
    map.addControl(new mapboxgl.ScaleControl({ unit: "metric" }), "bottom-right");

    // ── On style load ─────────────────────────────────────────────────────
    map.on("load", () => {
      // Atmosphere + stars
      map.setFog({
        color: "rgb(186, 210, 235)",
        "high-color": "rgb(36, 92, 223)",
        "horizon-blend": 0.02,
        "space-color": "rgb(11, 11, 25)",
        "star-intensity": 0.6,
      });

      // ── GeoJSON Source ─────────────────────────────────────────────────
      map.addSource("cities", {
        type: "geojson",
        data: buildGeoJSON(cities),
        // Enables feature-state for hover effects
        promoteId: "name",
      });

      // ── Layer 1: Outer glow ring (halo) ──────────────────────────────
      map.addLayer({
        id: "city-halo",
        type: "circle",
        source: "cities",
        paint: {
          "circle-radius": [
            "interpolate", ["linear"], ["zoom"],
            1, 10,
            4, 16,
            8, 22,
          ],
          "circle-color": [
            "match", ["get", "region_type"],
            "arid",      "#f59e0b",
            "temperate", "#22d3ee",
            "tropical",  "#4ade80",
            "#94a3b8",
          ],
          "circle-opacity": [
            "case",
            ["boolean", ["feature-state", "hover"], false], 0.35,
            0.15,
          ],
          "circle-blur": 0.6,
          "circle-pitch-alignment": "map",   // anchors to globe surface ✓
          "circle-pitch-scale": "map",
        },
      });

      // ── Layer 2: Main marker dot ──────────────────────────────────────
      map.addLayer({
        id: "city-points",
        type: "circle",
        source: "cities",
        paint: {
          // Slightly larger on hover via feature-state
          "circle-radius": [
            "case",
            ["boolean", ["feature-state", "hover"], false],
            [
              "interpolate", ["linear"], ["zoom"],
              1, 7,
              4, 10,
              8, 14,
            ],
            [
              "interpolate", ["linear"], ["zoom"],
              1, 5,
              4, 7,
              8, 10,
            ],
          ],
          "circle-color": [
            "match", ["get", "region_type"],
            "arid",      "#f59e0b",
            "temperate", "#22d3ee",
            "tropical",  "#4ade80",
            "#94a3b8",
          ],
          "circle-stroke-width": [
            "case",
            ["boolean", ["feature-state", "hover"], false], 2.5,
            1.5,
          ],
          "circle-stroke-color": "rgba(255,255,255,0.85)",
          "circle-opacity": 1,
          "circle-pitch-alignment": "map",   // locked to globe surface — NO DRIFT ✓
          "circle-pitch-scale": "map",
        },
      });

      // ── Layer 3: City name labels (visible at zoom ≥ 3) ─────────────
      map.addLayer({
        id: "city-labels",
        type: "symbol",
        source: "cities",
        minzoom: 3,
        layout: {
          "text-field": ["get", "name"],
          "text-font": ["DIN Offc Pro Medium", "Arial Unicode MS Bold"],
          "text-size": [
            "interpolate", ["linear"], ["zoom"],
            3, 10,
            6, 13,
          ],
          "text-offset": [0, 1.4],
          "text-anchor": "top",
          "text-allow-overlap": false,
          "text-ignore-placement": false,
        },
        paint: {
          "text-color": "#f1f5f9",
          "text-halo-color": "rgba(2,6,23,0.85)",
          "text-halo-width": 1.5,
        },
      });

      // ── Hover: cursor + feature-state ─────────────────────────────────
      map.on("mouseenter", "city-points", (e) => {
        map.getCanvas().style.cursor = "pointer";
        if (e.features.length > 0) {
          const id = e.features[0].properties.name;
          if (hoveredIdRef.current !== null && hoveredIdRef.current !== id) {
            map.setFeatureState({ source: "cities", id: hoveredIdRef.current }, { hover: false });
          }
          hoveredIdRef.current = id;
          map.setFeatureState({ source: "cities", id }, { hover: true });
        }
      });

      map.on("mouseleave", "city-points", () => {
        map.getCanvas().style.cursor = "";
        if (hoveredIdRef.current !== null) {
          map.setFeatureState({ source: "cities", id: hoveredIdRef.current }, { hover: false });
          hoveredIdRef.current = null;
        }
      });

      // ── Click: stable, no drift — WebGL layer event ───────────────────
      map.on("click", "city-points", (e) => {
        // Prevent map click from firing too
        e.originalEvent.stopPropagation();

        if (!e.features || e.features.length === 0) return;

        // Reconstruct the city object from GeoJSON properties
        const props = e.features[0].properties;
        const city = {
          name: props.name,
          lat:  Number(props.lat),
          lng:  Number(props.lng),
          temp: Number(props.temp),
          rain: Number(props.rain),
          humidity: Number(props.humidity),
          N: Number(props.N),
          P: Number(props.P),
          K: Number(props.K),
        };

        const coords = e.features[0].geometry.coordinates.slice();

        // Smooth fly-to
        map.flyTo({
          center: coords,
          zoom: Math.max(map.getZoom(), 3.5),
          duration: 900,
          essential: true,
        });

        // Show popup at exact GeoJSON coordinates (no drift)
        popupRef.current
          .setLngLat(coords)
          .setHTML(buildPopupHtml(props))
          .addTo(map);

        // ← Trigger existing ML pipeline (handleCityClick in App.jsx)
        onCitySelect(city);
      });

      // ── Prevent drag-rotate conflict on marker interaction ─────────────
      map.on("mousedown", "city-points", () => {
        map.dragRotate.disable();
        map.dragPan.disable();
      });
      map.on("mouseup", () => {
        map.dragRotate.enable();
        map.dragPan.enable();
      });

      // ── Start auto-rotation ────────────────────────────────────────────
      stopRotateRef.current = startAutoRotate(map);
    });

    return () => {
      if (stopRotateRef.current) stopRotateRef.current();
      if (popupRef.current) popupRef.current.remove();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", borderRadius: "inherit" }}
    />
  );
}
