/**
 * Climate Intelligence System — Dashboard Application
 * =====================================================
 * Interactive frontend connecting to FastAPI backend.
 * Renders charts with Plotly.js, manages tab navigation,
 * country search, policy simulation, and data fetching.
 */

const API_BASE = window.location.origin + '/api';

// ─── Plotly Theme ───
const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
    margin: { t: 20, r: 20, b: 40, l: 50 },
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        zerolinecolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 11 },
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        zerolinecolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 11 },
    },
    hoverlabel: {
        bgcolor: '#1e293b',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: { family: 'Inter', color: '#f0f4f8', size: 13 },
    },
    legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
};

const COLORS = {
    green: '#009E73',    // Okabe-Ito Bluish-Green
    blue: '#0072B2',     // Okabe-Ito Blue
    cyan: '#56B4E9',     // Okabe-Ito Sky Blue
    orange: '#E69F00',   // Okabe-Ito Orange
    red: '#D55E00',      // Okabe-Ito Vermillion
    purple: '#CC79A7',   // Okabe-Ito Red-Purple
    yellow: '#F0E442',   // Okabe-Ito Yellow
    teal: '#009E73',     // Re-map to Bluish-Green
    pink: '#CC79A7',     // Re-map to Red-Purple
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
};

// ─── State ───
let countriesList = [];
let selectedCountryISO = null;

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupCountrySearch();
    setupSimulation();
    loadGlobalOverview();
    loadCountriesList();
    setStatus('Connected', true);
});

// ─── Tab Navigation ───
function setupTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;

            // Update active states
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`content-${target}`).classList.add('active');

            // Load data for tab
            if (target === 'risk') loadRiskIntelligence();
            if (target === 'simulate') loadSimulationCountries();
        });
    });
}

// ─── Status ───
function setStatus(text, connected) {
    document.getElementById('status-text').textContent = text;
    const dot = document.querySelector('.status-dot');
    dot.style.background = connected ? '#06d6a0' : '#ef4444';
}

// ─── API Fetch Helper ───
async function fetchAPI(endpoint) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API error [${endpoint}]:`, err);
        return null;
    }
}

async function postAPI(endpoint, body) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API error [${endpoint}]:`, err);
        return null;
    }
}

// ═══════════════════════════════════════════════════
// TAB 1: GLOBAL OVERVIEW
// ═══════════════════════════════════════════════════
async function loadGlobalOverview() {
    const data = await fetchAPI('/global-overview');
    if (!data) { setStatus('Error loading data', false); return; }

    // KPI Cards
    const kpis = data.kpis;
    animateValue('kpi-total-co2', kpis.total_co2_gt, 1, ' Gt');
    animateValue('kpi-temp-anomaly', kpis.avg_temperature_anomaly, 3, '°C');
    animateValue('kpi-renewables-val', kpis.avg_renewables_share, 1, '%');
    animateValue('kpi-avg-risk', kpis.avg_risk_score, 1, '/100');

    // Emissions trend chart
    renderEmissionsTrend(data.emissions_trend);

    // Top emitters
    renderTopEmitters(data.top_emitters);

    // World map (needs country-level data)
    loadWorldMap();

    setStatus(`Data: ${data.latest_year} · ${data.total_countries} countries`, true);
}

function animateValue(elemId, target, decimals, suffix = '') {
    const el = document.getElementById(elemId);
    if (!el || target == null) return;

    const duration = 1200;
    const steps = 40;
    const stepTime = duration / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        const progress = step / steps;
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        current = target * eased;
        el.textContent = current.toFixed(decimals) + suffix;

        if (step >= steps) {
            el.textContent = target.toFixed(decimals) + suffix;
            clearInterval(timer);
        }
    }, stepTime);
}

function renderEmissionsTrend(trend) {
    if (!trend || !trend.length) return;

    const years = trend.map(d => d.year);
    const co2 = trend.map(d => d.total_co2 ? d.total_co2 / 1000 : null); // Mt → Gt

    const traces = [{
        x: years,
        y: co2,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: 'rgba(0, 158, 115, 0.08)',
        line: { color: '#009E73', width: 2.5, shape: 'spline' },
        name: 'Global CO₂ (Gt)',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        yaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.yaxis)), title: { text: 'CO₂ (Gt)', font: { size: 11 } } },
        showlegend: false,
    };

    Plotly.newPlot('chart-emissions-trend', traces, layout, PLOTLY_CONFIG);
}

function renderTopEmitters(emitters) {
    if (!emitters || !emitters.length) return;

    const sorted = [...emitters].sort((a, b) => a.co2 - b.co2);
    const names = sorted.map(d => d.country);
    const values = sorted.map(d => d.co2);

    const traces = [{
        x: values,
        y: names,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: values.map((_, i) => {
                const ratio = i / values.length;
                return `rgba(0, 158, 115, ${0.3 + ratio * 0.7})`;
            }),
            line: { width: 0 },
        },
        text: values.map(v => `${(v / 1000).toFixed(1)} Gt`),
        textposition: 'outside',
        textfont: { color: '#94a3b8', size: 10 },
        hovertemplate: '%{y}: %{x:.0f} Mt CO₂<extra></extra>',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.margin)), l: 100, r: 80 },
        xaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.xaxis)), title: { text: 'CO₂ (Mt)', font: { size: 11 } } },
        showlegend: false,
    };

    Plotly.newPlot('chart-top-emitters', traces, layout, PLOTLY_CONFIG);
}

async function loadWorldMap() {
    const data = await fetchAPI('/countries');
    if (!data) return;

    const metric = document.getElementById('map-metric').value;
    renderWorldMap(data.countries, metric);

    // Re-render on metric change
    document.getElementById('map-metric').onchange = () => {
        renderWorldMap(data.countries, document.getElementById('map-metric').value);
    };
}

function renderWorldMap(countries, metric) {
    const isos = countries.map(c => c.iso_code);
    const values = countries.map(c => c[metric] || 0);
    const names = countries.map(c => c.country);

    const colorscaleMap = {
        co2_per_capita: [[0, '#0a2f1f'], [0.3, '#009E73'], [0.6, '#E69F00'], [1, '#D55E00']],
        coal_share: [[0, '#009E73'], [0.5, '#E69F00'], [1, '#D55E00']],
        renewables_share: [[0, '#D55E00'], [0.5, '#E69F00'], [1, '#009E73']],
        risk_score: [[0, '#009E73'], [0.5, '#E69F00'], [1, '#D55E00']],
    };

    const titleMap = {
        co2_per_capita: 'CO₂ per Capita (tonnes)',
        risk_score: 'Risk Score (0–100)',
        renewables_share: 'Renewables Share (%)',
        coal_share: 'Coal Share (%)',
    };

    const traces = [{
        type: 'choropleth',
        locations: isos,
        z: values,
        text: names,
        colorscale: colorscaleMap[metric] || colorscaleMap.co2_per_capita,
        reversescale: false,
        marker: { line: { color: 'rgba(255,255,255,0.1)', width: 0.5 } },
        colorbar: {
            title: { text: titleMap[metric] || metric, font: { size: 11 } },
            thickness: 12,
            len: 0.5,
            tickfont: { size: 10 },
        },
        hovertemplate: '<b>%{text}</b><br>' + (titleMap[metric] || metric) + ': %{z:.1f}<extra></extra>',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { t: 10, r: 10, b: 10, l: 10 },
        geo: {
            bgcolor: 'rgba(0,0,0,0)',
            showframe: false,
            showcoastlines: true,
            coastlinecolor: 'rgba(255,255,255,0.1)',
            showland: true,
            landcolor: 'rgba(255,255,255,0.03)',
            showocean: true,
            oceancolor: 'rgba(10, 14, 23, 1)',
            showlakes: false,
            showcountries: true,
            countrycolor: 'rgba(255,255,255,0.05)',
            projection: { type: 'natural earth' },
        },
    };

    Plotly.newPlot('chart-world-map', traces, layout, PLOTLY_CONFIG);
}

// ═══════════════════════════════════════════════════
// TAB 2: COUNTRY DEEP DIVE
// ═══════════════════════════════════════════════════
async function loadCountriesList() {
    const data = await fetchAPI('/countries');
    if (!data) return;
    countriesList = data.countries;
}

function setupCountrySearch() {
    const input = document.getElementById('country-search');
    const dropdown = document.getElementById('country-dropdown');

    input.addEventListener('input', () => {
        const query = input.value.toLowerCase().trim();
        dropdown.innerHTML = '';

        if (query.length < 1) {
            dropdown.classList.remove('show');
            return;
        }

        const matches = countriesList.filter(c =>
            c.country?.toLowerCase().includes(query) ||
            c.iso_code?.toLowerCase().includes(query)
        ).slice(0, 15);

        if (matches.length === 0) {
            dropdown.classList.remove('show');
            return;
        }

        matches.forEach(c => {
            const item = document.createElement('div');
            item.className = 'search-item';
            item.innerHTML = `<span>${c.country}</span><span class="iso">${c.iso_code}</span>`;
            item.addEventListener('click', () => {
                selectCountry(c.iso_code, c.country);
                dropdown.classList.remove('show');
                input.value = '';
            });
            dropdown.appendChild(item);
        });

        dropdown.classList.add('show');
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-wrapper')) {
            dropdown.classList.remove('show');
        }
    });
}

async function selectCountry(iso, name) {
    selectedCountryISO = iso;
    document.getElementById('country-name-display').textContent = name;
    document.getElementById('country-iso-display').textContent = iso;

    // Load all country data
    const [countryData, forecastData, shapData, recData] = await Promise.all([
        fetchAPI(`/country/${iso}`),
        fetchAPI(`/forecasts/${iso}`),
        fetchAPI(`/shap/${iso}`),
        fetchAPI(`/recommendations/${iso}`),
    ]);

    if (countryData) {
        const latest = countryData.latest;
        updateCountryKPIs(latest);
        renderEnergyMix(latest);
        renderRiskGauge(latest.risk_score || 0, name);
    }

    if (forecastData) {
        renderCountryForecast(forecastData, name);
    }

    if (shapData) {
        renderCountrySHAP(shapData);
    }

    if (recData) {
        renderRecommendations(recData);
    }
}

function updateCountryKPIs(latest) {
    const set = (id, val, dec = 1) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val != null ? Number(val).toFixed(dec) : '—';
    };
    set('ckpi-co2', latest.co2_per_capita, 2);
    set('ckpi-risk', latest.risk_score, 1);
    set('ckpi-renewables', latest.renewables_share, 1);
    set('ckpi-coal', latest.coal_share, 1);
}

function renderCountryForecast(data, name) {
    const traces = [];

    // Historical
    if (data.historical && data.historical.length) {
        traces.push({
            x: data.historical.map(d => Number(d.year)),
            y: data.historical.map(d => Number(d.co2)),
            type: 'scatter',
            mode: 'lines',
            name: 'Historical',
            line: { color: COLORS.blue, width: 2.5 },
        });
    }

    // Forecast
    if (data.forecast && data.forecast.length) {
        const fcYears = data.forecast.map(d => Number(d.year));
        const fcValues = data.forecast.map(d => Number(d.forecast_co2));
        const fcUpper = data.forecast.map(d => Number(d.forecast_co2_upper));
        const fcLower = data.forecast.map(d => Number(d.forecast_co2_lower));

        // Connect historical to forecast
        if (data.historical && data.historical.length) {
            const lastHist = data.historical[data.historical.length - 1];
            fcYears.unshift(lastHist.year);
            fcValues.unshift(lastHist.co2);
            if (fcUpper.length) fcUpper.unshift(lastHist.co2);
            if (fcLower.length) fcLower.unshift(lastHist.co2);
        }

        // Confidence interval
        traces.push({
            x: [...fcYears, ...fcYears.slice().reverse()],
            y: [...fcUpper, ...fcLower.slice().reverse()],
            type: 'scatter',
            fill: 'toself',
            fillcolor: 'rgba(6, 214, 160, 0.08)',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip',
        });

        traces.push({
            x: fcYears,
            y: fcValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Forecast',
            line: { color: COLORS.green, width: 2.5, dash: 'dot' },
        });

        const badge = document.getElementById('forecast-badge');
        if (badge) {
            badge.textContent = data.forecast[0].model_used || 'Forecast';
            badge.title = 'Current modeling pipeline used for this forecast';
        }
    }

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        yaxis: {
            ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.yaxis)),
            type: 'linear',
            title: { text: 'CO₂ Emissions (Mt)', font: { size: 11 } },
            tickformat: '.0f'
        },
    };

    Plotly.newPlot('chart-country-forecast', traces, layout, PLOTLY_CONFIG);
}

function renderRiskGauge(score, name) {
    const color = score > 70 ? COLORS.red : score > 40 ? COLORS.orange : COLORS.green;

    const traces = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: score,
        number: { font: { size: 42, color: color, family: 'JetBrains Mono' }, suffix: '' },
        gauge: {
            axis: { range: [0, 100], tickwidth: 1, tickcolor: 'rgba(255,255,255,0.1)', dtick: 20 },
            bar: { color: color, thickness: 0.7 },
            bgcolor: 'rgba(255,255,255,0.03)',
            borderwidth: 0,
            steps: [
                { range: [0, 30], color: 'rgba(6, 214, 160, 0.08)' },
                { range: [30, 60], color: 'rgba(245, 158, 11, 0.08)' },
                { range: [60, 100], color: 'rgba(239, 68, 68, 0.08)' },
            ],
        },
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { t: 30, r: 30, b: 10, l: 30 },
    };

    Plotly.newPlot('chart-risk-gauge', traces, layout, PLOTLY_CONFIG);
}

function renderEnergyMix(latest) {
    const labels = ['Coal', 'Gas', 'Oil', 'Renewables/Other'];
    const values = [
        latest.coal_share || 0,
        latest.gas_share || 0,
        latest.oil_share || 0,
        latest.renewables_share || 0,
    ];

    const traces = [{
        labels: labels,
        values: values,
        type: 'pie',
        hole: 0.55,
        marker: {
            colors: [COLORS.red, COLORS.orange, COLORS.purple, COLORS.green],
            line: { color: 'rgba(10,14,23,1)', width: 2 },
        },
        textinfo: 'label+percent',
        textfont: { size: 11, color: '#f0f4f8' },
        hovertemplate: '%{label}: %{percent}<extra></extra>',
        sort: false,
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { t: 20, r: 20, b: 20, l: 20 },
        showlegend: false,
    };

    Plotly.newPlot('chart-energy-mix', traces, layout, PLOTLY_CONFIG);
}

function renderCountrySHAP(shapData) {
    const shapValues = shapData.shap_values || {};
    const entries = Object.entries(shapValues).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    if (entries.length === 0) {
        document.getElementById('chart-country-shap').innerHTML = '<p class="placeholder-text">No SHAP data available</p>';
        return;
    }

    const features = entries.map(([k]) => k.replace(/_/g, ' '));
    const values = entries.map(([, v]) => v);
    const colors = values.map(v => v > 0 ? COLORS.red : COLORS.green);

    const traces = [{
        x: values,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: { color: colors },
        hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.margin)), l: 140 },
        xaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.xaxis)), title: { text: 'SHAP Value (impact on risk)', font: { size: 11 } } },
        showlegend: false,
    };

    Plotly.newPlot('chart-country-shap', traces, layout, PLOTLY_CONFIG);
}

function renderRecommendations(recData) {
    const container = document.getElementById('recommendations-list');
    const recs = recData.recommendations || [];

    if (recs.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No recommendations available</p>';
        return;
    }

    container.innerHTML = recs.map(r =>
        `<div class="rec-item">
            <span class="rec-icon">▸</span>
            <span>${r}</span>
        </div>`
    ).join('');
}

// ═══════════════════════════════════════════════════
// TAB 3: RISK INTELLIGENCE
// ═══════════════════════════════════════════════════
async function loadRiskIntelligence() {
    const [riskData, clusterData, shapData] = await Promise.all([
        fetchAPI('/risk-scores'),
        fetchAPI('/clusters'),
        fetchAPI('/shap/USA'), // Get feature importance from any country
    ]);

    if (riskData) {
        renderRiskRanking(riskData.scores);
        renderRiskMap(riskData.scores);
    }

    if (clusterData) {
        renderClusters(clusterData);
        renderClusterTable(clusterData);
    }

    // Feature importance
    loadFeatureImportance();
}

async function loadFeatureImportance() {
    // Try to load feature importance directly
    try {
        const res = await fetch(`${API_BASE}/shap/USA`);
        const data = await res.json();
        if (data.feature_importance && data.feature_importance.length) {
            renderFeatureImportance(data.feature_importance);
        }
    } catch (e) {
        console.log('Feature importance not available');
    }
}

function renderRiskRanking(scores) {
    if (!scores || !scores.length) return;

    const top15 = scores.slice(0, 15);
    const sorted = [...top15].reverse();

    const traces = [{
        x: sorted.map(d => d.risk_score || 0),
        y: sorted.map(d => d.country),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(d => {
                const s = d.risk_score || 0;
                if (s > 70) return COLORS.red;
                if (s > 50) return COLORS.orange;
                return COLORS.green;
            }),
        },
        text: sorted.map(d => `${(d.risk_score || 0).toFixed(1)}`),
        textposition: 'outside',
        textfont: { color: '#94a3b8', size: 10, family: 'JetBrains Mono' },
        hovertemplate: '<b>%{y}</b><br>Risk Score: %{x:.1f}<extra></extra>',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.margin)), l: 120, r: 50 },
        xaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.xaxis)), range: [0, 110], title: { text: 'Risk Score', font: { size: 11 } } },
        showlegend: false,
    };

    Plotly.newPlot('chart-risk-ranking', traces, layout, PLOTLY_CONFIG);
}

function renderFeatureImportance(features) {
    if (!features || !features.length) return;

    const sorted = [...features].sort((a, b) => a.importance - b.importance);

    const traces = [{
        x: sorted.map(d => d.importance),
        y: sorted.map(d => d.feature.replace(/_/g, ' ')),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map((_, i) => {
                const colors = [COLORS.cyan, COLORS.blue, COLORS.green, COLORS.teal, COLORS.purple, COLORS.orange, COLORS.pink, COLORS.red];
                return colors[i % colors.length];
            }),
        },
        hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.margin)), l: 150 },
        xaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.xaxis)), title: { text: 'Mean |SHAP|', font: { size: 11 } } },
        showlegend: false,
    };

    Plotly.newPlot('chart-feature-importance', traces, layout, PLOTLY_CONFIG);
}

function renderClusters(clusterData) {
    const clusters = clusterData.clusters || [];
    if (!clusters.length) return;

    const clusterColors = {
        'High Emitters': COLORS.red,
        'Transition Economies': COLORS.orange,
        'Green Economies': COLORS.green,
    };

    const traces = [];
    const clusterNames = [...new Set(clusters.map(c => c.cluster_name))];

    clusterNames.forEach(name => {
        const subset = clusters.filter(c => c.cluster_name === name);
        traces.push({
            x: subset.map(d => d.co2_per_capita || 0),
            y: subset.map(d => d.renewables_share || 0),
            text: subset.map(d => d.country),
            type: 'scatter',
            mode: 'markers',
            name: name,
            marker: {
                color: clusterColors[name] || COLORS.blue,
                size: 10,
                opacity: 0.8,
                line: { width: 1, color: 'rgba(255,255,255,0.2)' },
            },
            hovertemplate: '<b>%{text}</b><br>CO₂/cap: %{x:.1f}<br>Renewables: %{y:.1f}%<extra>' + name + '</extra>',
        });
    });

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        xaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.xaxis)), title: { text: 'CO₂ per Capita (t)', font: { size: 11 } } },
        yaxis: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.yaxis)), title: { text: 'Renewables Share (%)', font: { size: 11 } }, tickformat: '.1f' },
        legend: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.legend)), y: 1.1, x: 0 },
    };

    Plotly.newPlot('chart-clusters', traces, layout, PLOTLY_CONFIG);
}

function renderClusterTable(clusterData) {
    const clusters = clusterData.clusters || [];
    if (!clusters.length) return;

    // Sort by risk-relevant metrics
    const sorted = [...clusters].sort((a, b) => (b.co2_per_capita || 0) - (a.co2_per_capita || 0));

    let html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Country</th>
                    <th>Cluster</th>
                    <th>CO₂/Cap</th>
                    <th>Renewables</th>
                </tr>
            </thead>
            <tbody>
    `;

    sorted.forEach(c => {
        const badgeClass = c.cluster_name === 'High Emitters' ? 'high' :
            c.cluster_name === 'Transition Economies' ? 'medium' : 'low';
        html += `
            <tr>
                <td>${c.country || c.iso_code}</td>
                <td><span class="risk-badge ${badgeClass}">${c.cluster_name || '—'}</span></td>
                <td>${(c.co2_per_capita || 0).toFixed(1)}</td>
                <td>${(c.renewables_share || 0).toFixed(1)}%</td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    document.getElementById('cluster-table').innerHTML = html;
}

function renderRiskMap(scores) {
    if (!scores || !scores.length) return;

    const isos = scores.map(d => d.iso_code);
    const riskValues = scores.map(d => d.risk_score || 0);
    const names = scores.map(d => d.country);

    const traces = [{
        type: 'choropleth',
        locations: isos,
        z: riskValues,
        text: names,
        colorscale: [[0, COLORS.green], [0.3, COLORS.cyan], [0.5, COLORS.orange], [0.7, COLORS.red], [1, COLORS.red]],
        marker: { line: { color: 'rgba(255,255,255,0.1)', width: 0.5 } },
        colorbar: {
            title: { text: 'Risk Score', font: { size: 11 } },
            thickness: 12,
            len: 0.5,
            tickfont: { size: 10 },
        },
        hovertemplate: '<b>%{text}</b><br>Risk Score: %{z:.1f}<extra></extra>',
    }];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        margin: { t: 10, r: 10, b: 10, l: 10 },
        geo: {
            bgcolor: 'rgba(0,0,0,0)',
            showframe: false,
            showcoastlines: true,
            coastlinecolor: 'rgba(255,255,255,0.1)',
            showland: true,
            landcolor: 'rgba(255,255,255,0.03)',
            showocean: true,
            oceancolor: 'rgba(10, 14, 23, 1)',
            showcountries: true,
            countrycolor: 'rgba(255,255,255,0.05)',
            projection: { type: 'natural earth' },
        },
    };

    Plotly.newPlot('chart-risk-map', traces, layout, PLOTLY_CONFIG);
}

// ═══════════════════════════════════════════════════
// TAB 4: POLICY SIMULATION
// ═══════════════════════════════════════════════════
async function loadSimulationCountries() {
    if (!countriesList.length) {
        const data = await fetchAPI('/countries');
        if (data) countriesList = data.countries;
    }

    const select = document.getElementById('sim-country');
    if (select.options.length <= 1) {
        select.innerHTML = '';
        const sorted = [...countriesList].sort((a, b) => (a.country || '').localeCompare(b.country || ''));
        sorted.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.iso_code;
            opt.textContent = `${c.country} (${c.iso_code})`;
            select.appendChild(opt);
        });
        // Default to India
        const indiaOpt = [...select.options].find(o => o.value === 'IND');
        if (indiaOpt) indiaOpt.selected = true;
    }
}

function setupSimulation() {
    const emSlider = document.getElementById('sim-emission-slider');
    const renSlider = document.getElementById('sim-renewables-slider');

    emSlider.addEventListener('input', () => {
        document.getElementById('sim-emission-val').textContent = emSlider.value + '%';
    });

    renSlider.addEventListener('input', () => {
        document.getElementById('sim-renewables-val').textContent = '+' + renSlider.value + 'pp';
    });

    document.getElementById('btn-simulate').addEventListener('click', runSimulation);
}

async function runSimulation() {
    const iso = document.getElementById('sim-country').value;
    const emReduction = parseInt(document.getElementById('sim-emission-slider').value);
    const renIncrease = parseInt(document.getElementById('sim-renewables-slider').value);

    const btn = document.getElementById('btn-simulate');
    btn.textContent = '⏳ Running...';
    btn.disabled = true;

    const result = await postAPI('/simulate', {
        iso_code: iso,
        emission_reduction_pct: emReduction,
        renewables_increase_pct: renIncrease,
    });

    btn.textContent = '🧪 Run Simulation';
    btn.disabled = false;

    if (!result) {
        alert('Simulation failed. Please try again.');
        return;
    }

    // Show results
    document.getElementById('sim-results').style.display = 'block';

    const orig = result.original;
    const sim = result.simulated;

    // Original values
    document.getElementById('sim-orig-risk').textContent = (orig.risk_score || 0).toFixed(1);
    document.getElementById('sim-orig-co2').textContent = (orig.co2_per_capita || 0).toFixed(2) + ' t';
    document.getElementById('sim-orig-ren').textContent = (orig.renewables_share || 0).toFixed(1) + '%';
    document.getElementById('sim-orig-coal').textContent = (orig.coal_share || 0).toFixed(1) + '%';

    // Simulated values
    document.getElementById('sim-new-risk').textContent = (sim.risk_score || 0).toFixed(1);
    document.getElementById('sim-new-co2').textContent = (sim.co2_per_capita || 0).toFixed(2) + ' t';
    document.getElementById('sim-new-ren').textContent = (sim.renewables_share || 0).toFixed(1) + '%';
    document.getElementById('sim-new-coal').textContent = (sim.coal_share || 0).toFixed(1) + '%';

    // Risk delta
    const delta = sim.risk_delta || 0;
    const deltaEl = document.getElementById('sim-risk-delta');
    deltaEl.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} Risk`;
    deltaEl.style.color = delta < 0 ? COLORS.green : COLORS.red;

    // Forecast comparison chart
    renderSimForecast(result.forecast_comparison, orig.country || iso);
}

function renderSimForecast(forecastComp, name) {
    if (!forecastComp || !forecastComp.length) {
        document.getElementById('chart-sim-forecast').innerHTML = '<p class="placeholder-text">No forecast data for comparison</p>';
        return;
    }

    const years = forecastComp.map(d => Number(d.year));
    const original = forecastComp.map(d => Number(d.forecast_co2_original));
    const simulated = forecastComp.map(d => Number(d.forecast_co2_simulated));

    const traces = [
        {
            x: years,
            y: original,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Business as Usual',
            line: { color: COLORS.red, width: 2, dash: 'dot' },
            marker: { size: 6 },
        },
        {
            x: years,
            y: simulated,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'With Policy',
            line: { color: COLORS.green, width: 2.5 },
            fill: 'tonexty',
            fillcolor: 'rgba(6, 214, 160, 0.08)',
            marker: { size: 6 },
        },
    ];

    const layout = {
        ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),
        yaxis: {
            ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.yaxis)),
            type: 'linear',
            title: { text: 'CO₂ Emissions (Mt)', font: { size: 11 } },
            tickformat: '.0f'
        },
        legend: { ...JSON.parse(JSON.stringify(PLOTLY_LAYOUT.legend)), y: 1.1, x: 0 },
    };

    Plotly.newPlot('chart-sim-forecast', traces, layout, PLOTLY_CONFIG);
}
