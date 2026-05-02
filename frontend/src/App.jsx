import { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = "http://127.0.0.1:8000";

const cities = [
  { name: "Lahore", lat: 31.5204, lng: 74.3587, rainfall: 620, temperature: 25.1, N: 88, P: 45, K: 42 },
  { name: "Karachi", lat: 24.8607, lng: 67.0011, rainfall: 180, temperature: 28.9, N: 65, P: 30, K: 36 },
  { name: "Islamabad", lat: 33.6844, lng: 73.0479, rainfall: 1150, temperature: 21.4, N: 74, P: 40, K: 39 },
  { name: "Rawalpindi", lat: 33.5651, lng: 73.0169, rainfall: 1080, temperature: 22.2, N: 78, P: 41, K: 40 },
  { name: "Faisalabad", lat: 31.4504, lng: 73.135, rainfall: 375, temperature: 25.9, N: 90, P: 48, K: 46 },
  { name: "Multan", lat: 30.1575, lng: 71.5249, rainfall: 185, temperature: 27.8, N: 84, P: 43, K: 41 },
  { name: "Peshawar", lat: 34.0151, lng: 71.5249, rainfall: 400, temperature: 23.7, N: 76, P: 38, K: 35 },
  { name: "Quetta", lat: 30.1798, lng: 66.975, rainfall: 260, temperature: 19.6, N: 64, P: 29, K: 30 },
  { name: "Hyderabad", lat: 25.396, lng: 68.3578, rainfall: 170, temperature: 29.4, N: 68, P: 34, K: 33 },
  { name: "Sialkot", lat: 32.4945, lng: 74.5229, rainfall: 910, temperature: 23.3, N: 82, P: 44, K: 42 },
  { name: "Gujranwala", lat: 32.1877, lng: 74.1945, rainfall: 780, temperature: 24.1, N: 80, P: 42, K: 41 },
  { name: "Bahawalpur", lat: 29.3956, lng: 71.6836, rainfall: 145, temperature: 28.2, N: 72, P: 35, K: 36 }
];

const markerIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34]
});

async function postJson(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

function Spinner() {
  return (
    <div className="spinner-wrap">
      <div className="spinner" />
      <p>Fetching live analytics...</p>
    </div>
  );
}

export default function App() {
  const [selectedCity, setSelectedCity] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [modelHealth, setModelHealth] = useState(null);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [metricsRefreshedAt, setMetricsRefreshedAt] = useState("");

  const fetchModelHealth = async () => {
    try {
      const [metricsRes, historyRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/metrics/history`)
      ]);
      if (!metricsRes.ok) {
        throw new Error(`metrics failed: ${metricsRes.status}`);
      }
      const data = await metricsRes.json();
      setModelHealth(data.artifacts ?? null);
      setMetricsRefreshedAt(data.generated_at ?? new Date().toISOString());
      if (historyRes.ok) {
        const h = await historyRes.json();
        setMetricsHistory(Array.isArray(h.history) ? h.history : []);
      } else {
        setMetricsHistory([]);
      }
    } catch {
      setModelHealth(null);
      setMetricsHistory([]);
      setMetricsRefreshedAt("");
    }
  };

  useEffect(() => {
    fetchModelHealth();
  }, []);

  const getStatusLabel = (entry) => {
    if (!entry) return "Unknown";
    if (entry.available) return "Online";
    if (entry.error) return "Error";
    return "Missing";
  };

  const getStatusClass = (entry) => {
    if (!entry) return "status-unknown";
    if (entry.available) return "status-online";
    if (entry.error) return "status-error";
    return "status-missing";
  };

  const metricText = (metrics) => {
    if (!metrics) return "No metrics";
    if (typeof metrics.holdout_accuracy === "number") {
      return `Acc ${metrics.holdout_accuracy.toFixed(2)} | F1 ${(metrics.holdout_f1_weighted ?? 0).toFixed(2)}`;
    }
    if (typeof metrics.accuracy === "number") {
      return `Acc ${metrics.accuracy.toFixed(2)} | F1 ${(metrics.f1_score ?? 0).toFixed(2)} | RMSE ${(metrics.rmse ?? 0).toFixed(2)}`;
    }
    return "Metrics tracked";
  };

  const metricValueForTrend = (metrics) => {
    if (!metrics) return { key: "", value: null, direction: "neutral" };
    if (typeof metrics.rmse === "number") {
      return { key: "rmse", value: metrics.rmse, direction: "lower_better" };
    }
    if (typeof metrics.holdout_rmse === "number") {
      return { key: "holdout_rmse", value: metrics.holdout_rmse, direction: "lower_better" };
    }
    if (typeof metrics.holdout_accuracy === "number") {
      return { key: "holdout_accuracy", value: metrics.holdout_accuracy, direction: "higher_better" };
    }
    if (typeof metrics.accuracy === "number") {
      return { key: "accuracy", value: metrics.accuracy, direction: "higher_better" };
    }
    return { key: "", value: null, direction: "neutral" };
  };

  const trendForModel = (modelKey, currentMetrics) => {
    const last = metricsHistory.at(-1)?.artifacts?.[modelKey]?.metrics;
    const prev = metricsHistory.at(-2)?.artifacts?.[modelKey]?.metrics;
    const current = metricValueForTrend(currentMetrics);
    const previous = metricValueForTrend(prev ?? last);
    if (!current.key || current.value == null || previous.value == null) {
      return { symbol: "•", label: "No trend", className: "trend-flat" };
    }
    const delta = Number(current.value) - Number(previous.value);
    if (Math.abs(delta) < 1e-9) {
      return { symbol: "→", label: "Stable", className: "trend-flat" };
    }
    if (current.direction === "lower_better") {
      return delta < 0
        ? { symbol: "↓", label: "Improving", className: "trend-up" }
        : { symbol: "↑", label: "Regressing", className: "trend-down" };
    }
    return delta > 0
      ? { symbol: "↑", label: "Improving", className: "trend-up" }
      : { symbol: "↓", label: "Regressing", className: "trend-down" };
  };

  const handleCityClick = async (city) => {
    setSelectedCity(city);
    setLoading(true);
    setError("");

    const baseFeatures = {
      rainfall: city.rainfall,
      temperature: city.temperature,
      N: city.N,
      P: city.P,
      K: city.K
    };

    try {
      const [yieldResp, classResp, forecastResp, recommendResp] = await Promise.all([
        postJson("/predict-yield", { features: baseFeatures }),
        postJson("/classify-yield", { features: baseFeatures }),
        postJson("/forecast", { periods: 12, context_features: baseFeatures }),
        postJson("/recommend", { ...baseFeatures })
      ]);

      let clusterResp = null;
      try {
        clusterResp = await postJson("/cluster", { samples: [baseFeatures] });
      } catch (clusterErr) {
        clusterResp = { clusters: [], error: String(clusterErr) };
      }

      setResult({
        cropSuitabilityScore: yieldResp.crop_suitability_score ?? yieldResp.prediction,
        riskLevel: classResp.risk_level ?? classResp.prediction,
        projectedSuitability: forecastResp.suitability_projection ?? null,
        rainfallHistory: forecastResp.historical ?? [],
        forecast: forecastResp.forecast ?? [],
        recommendation: recommendResp,
        cluster: clusterResp
      });
      await fetchModelHealth();
    } catch (err) {
      setError(err.message || "Unable to fetch data from backend.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    const history = result?.rainfallHistory ?? [];
    const points = result?.forecast ?? [];
    const historyLabels = history.map((p) => p.date);
    const forecastLabels = points.map((p) => p.date);
    const labels = [...historyLabels, ...forecastLabels];
    const historyData = [...history.map((p) => p.rainfall), ...forecastLabels.map(() => null)];
    const forecastData = [...historyLabels.map(() => null), ...points.map((p) => p.rainfall ?? p.prediction)];
    return {
      labels,
      datasets: [
        {
          label: "Historical Rainfall",
          data: historyData,
          borderColor: "#94a3b8",
          backgroundColor: "rgba(148,163,184,0.15)",
          tension: 0.2,
          pointRadius: 1.5,
          fill: false
        },
        {
          label: "Forecast Rainfall",
          data: forecastData,
          borderColor: "#22d3ee",
          backgroundColor: "rgba(34,211,238,0.2)",
          tension: 0.3,
          pointRadius: 2,
          borderDash: [6, 3],
          fill: false
        }
      ]
    };
  }, [result]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#c7d2fe" } }
    },
    scales: {
      x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.08)" } },
      y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.08)" }, title: { display: true, text: "Rainfall", color: "#cbd5e1" } }
    }
  };

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="panel glass">
          <h1>Agri Intelligence</h1>
          <p className="muted">Pakistan crop insights powered by ML + geospatial mapping.</p>
        </div>

        <div className={`panel glass transition ${loading ? "dimmed" : ""}`}>
          {!selectedCity && <p className="muted">Click any city marker to load analytics.</p>}
          {loading && <Spinner />}
          {!loading && error && <div className="error-box">{error}</div>}

          {!loading && !error && result && selectedCity && (
            <div className="metrics premium">
              <h2>{selectedCity.name}</h2>
              <div className="metric-grid">
                <div className="metric-card score-card">
                  <span>Crop Suitability Score</span>
                  <strong>{Number(result.cropSuitabilityScore || 0).toFixed(2)}</strong>
                </div>
                <div className="metric-card risk-card">
                  <span>Risk Level</span>
                  <strong>{result.riskLevel || "N/A"}</strong>
                </div>
                <div className="metric-card">
                  <span>Cluster</span>
                  <strong>{result.cluster?.clusters?.[0] ?? "N/A"}</strong>
                </div>
              </div>

              {result.projectedSuitability && (
                <div className="projection-box">
                  <p>
                    <strong>Projected next-month score:</strong>{" "}
                    {Number(result.projectedSuitability.crop_suitability_score || 0).toFixed(2)}
                  </p>
                  <p>
                    <strong>Projected risk:</strong> {result.projectedSuitability.risk_level || "N/A"}
                  </p>
                </div>
              )}

              <div className="recommendation">
                <h3>Recommendation</h3>
                <p><strong>Best crop:</strong> {result.recommendation?.best_crop || "N/A"}</p>
                <p>{result.recommendation?.fertilizer_suggestion || "No fertilizer suggestion."}</p>
              </div>

              <div className="health-box">
                <div className="health-header">
                  <h3>Model Health</h3>
                  <button className="refresh-btn" onClick={fetchModelHealth} type="button">
                    Refresh
                  </button>
                </div>
                {metricsRefreshedAt && (
                  <p className="muted health-time">
                    Last updated: {new Date(metricsRefreshedAt).toLocaleString()}
                  </p>
                )}
                {!modelHealth && <p className="muted">Model metrics unavailable.</p>}
                {modelHealth && (
                  <>
                    {Object.entries(modelHealth).map(([name, entry]) => (
                      <div className="health-row" key={name}>
                        <span>{name.replaceAll("_", " ")}</span>
                        <strong className={`status-badge ${getStatusClass(entry)}`}>
                          {getStatusLabel(entry)}
                        </strong>
                        <em>{metricText(entry?.metrics)}</em>
                        <small className={`trend-chip ${trendForModel(name, entry?.metrics).className}`}>
                          {trendForModel(name, entry?.metrics).symbol} {trendForModel(name, entry?.metrics).label}
                        </small>
                      </div>
                    ))}
                  </>
                )}
              </div>

              <div className="chart-wrap">
                <h3>Rainfall Forecast</h3>
                {result.forecast.length > 0 ? (
                  <div className="chart-container">
                    <Line data={chartData} options={chartOptions} />
                  </div>
                ) : (
                  <p className="muted">No forecast data available.</p>
                )}
              </div>
            </div>
          )}
        </div>
      </aside>

      <main className="map-area">
        <MapContainer center={[30.3753, 69.3451]} zoom={5.2} scrollWheelZoom className="map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {cities.map((city) => (
            <Marker
              key={city.name}
              position={[city.lat, city.lng]}
              icon={markerIcon}
              eventHandlers={{ click: () => handleCityClick(city) }}
            >
              <Popup>
                <strong>{city.name}</strong>
                <br />
                Rainfall: {city.rainfall} mm
                <br />
                Temp: {city.temperature} C
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </main>
    </div>
  );
}
