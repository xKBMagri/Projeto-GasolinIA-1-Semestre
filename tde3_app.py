# ============================================================
# TDE 3 — Raciocínio Matemático para Computação
# Previsão de Consumo de Combustível
# 
# Como rodar:
#   1. pip install flask scikit-learn pandas numpy
#   2. python tde3_app.py
#   3. Abra http://localhost:5000 no navegador
# ============================================================

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ============================================================
# DADOS E TREINAMENTO DO MODELO
# ============================================================
dados = {
    'Cilindrada_L':  [1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.6, 1.8, 2.0, 2.0, 2.0, 2.8, 3.0, 5.0, 1.3],
    'Peso_kg':       [870, 940, 865, 975, 1020, 1200, 1150, 1440, 1345, 1380, 1260, 2120, 1730, 2200, 1100],
    'Potencia_cv':   [65, 75, 75, 80, 82, 109, 120, 139, 154, 177, 132, 177, 510, 400, 95],
    'Aceleracao_s':  [15.0, 14.0, 13.5, 13.0, 12.5, 11.0, 11.5, 9.5, 8.5, 8.0, 9.2, 11.0, 3.9, 6.5, 12.0],
    'Ano':           [2010, 2012, 2020, 2019, 2021, 2022, 2018, 2021, 2022, 2023, 2020, 2020, 2023, 2022, 2021],
    'Consumo_kmL':   [14.2, 13.5, 14.8, 13.0, 12.8, 11.5, 11.8, 9.5, 10.5, 10.0, 11.2, 7.2, 6.2, 5.8, 12.5],
}
nomes = [
    'Fiat Uno', 'VW Gol', 'Fiat Mobi', 'HB20', 'Onix',
    'Fiat Strada', 'VW Polo', 'Jeep Renegade', 'Honda Civic',
    'Toyota Corolla', 'VW Jetta', 'Toyota SW4', 'BMW M3', 'Ford F-150', 'VW Up'
]

df = pd.DataFrame(dados, index=nomes)
X = df[['Cilindrada_L', 'Peso_kg', 'Potencia_cv', 'Aceleracao_s', 'Ano']].values
y = df['Consumo_kmL'].values

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

r2   = r2_score(y_treino, modelo.predict(X_treino))
rmse = np.sqrt(mean_squared_error(y_teste, modelo.predict(X_teste)))
print(f"✅ Modelo treinado! R² = {r2:.4f} | RMSE = {rmse:.4f} km/L")

# ============================================================
# ROTA DE PREVISÃO (API)
# ============================================================
@app.route('/prever', methods=['POST'])
def prever():
    try:
        d = request.json
        entrada = np.array([[
            float(d['cilindrada']),
            float(d['peso']),
            float(d['potencia']),
            float(d['aceleracao']),
            float(d['ano'])
        ]])
        consumo = float(modelo.predict(entrada)[0])
        consumo = round(max(2.0, min(22.0, consumo)), 2)
        return jsonify({
            'consumo':   consumo,
            'l100':      round(100 / consumo, 2),
            'custo':     round((100 / consumo) * 5.90, 2),
            'autonomia': round(consumo * 50),
            'r2':        round(r2, 4),
            'rmse':      round(rmse, 4),
        })
    except Exception as e:
        return jsonify({'erro': str(e)}), 400

# ============================================================
# PÁGINA HTML (embutida no Python)
# ============================================================
HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TDE3 — Previsão de Consumo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e8eaf0; min-height: 100vh; padding: 2rem 1rem; }
  .container { max-width: 700px; margin: 0 auto; }
  h1 { font-size: 1.4rem; font-weight: 600; color: #fff; margin-bottom: .25rem; }
  .sub { font-size: .78rem; color: #6b7280; margin-bottom: 2rem; text-transform: uppercase; letter-spacing: .05em; }
  .tag { font-size: .65rem; background: #1e1b4b; color: #a5b4fc; padding: .2rem .6rem; border-radius: 6px; margin-left: .5rem; vertical-align: middle; }
  .card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.25rem; }
  .card-title { font-size: .7rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: #6b7280; margin-bottom: 1.1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .field label { font-size: .78rem; color: #9ca3af; display: block; margin-bottom: .4rem; }
  .field input { width: 100%; background: #0f1117; border: 1px solid #2a2d3a; border-radius: 8px; padding: .55rem .75rem; color: #e8eaf0; font-size: .9rem; outline: none; transition: border .2s; }
  .field input:focus { border-color: #6366f1; }
  .field .hint { font-size: .68rem; color: #4b5563; margin-top: .3rem; }
  button.predict { width: 100%; background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; border-radius: 10px; padding: .85rem; color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: .75rem; transition: opacity .2s; }
  button.predict:hover { opacity: .88; }
  button.predict:disabled { opacity: .5; cursor: not-allowed; }
  .result-card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 14px; padding: 1.75rem; text-align: center; display: none; }
  .result-card.show { display: block; }
  .result-num { font-size: 3.5rem; font-weight: 700; line-height: 1; }
  .result-unit { font-size: 1rem; color: #9ca3af; margin-top: .3rem; }
  .badge { display: inline-block; font-size: .75rem; font-weight: 600; padding: .3rem .9rem; border-radius: 20px; margin-top: .75rem; }
  .badge-green { background: #064e3b; color: #6ee7b7; }
  .badge-yellow { background: #451a03; color: #fcd34d; }
  .badge-red { background: #450a0a; color: #fca5a5; }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin-top: 1.25rem; }
  .metric { background: #0f1117; border-radius: 10px; padding: .85rem; text-align: center; }
  .metric-val { font-size: 1.3rem; font-weight: 600; }
  .metric-lbl { font-size: .7rem; color: #6b7280; margin-top: .2rem; }
  .bar-wrap { margin-top: 1.25rem; }
  .bar-lbl { display: flex; justify-content: space-between; font-size: .72rem; color: #6b7280; margin-bottom: .35rem; }
  .bar-bg { background: #0f1117; border-radius: 99px; height: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 99px; transition: width .6s ease; }
  .error { background: #1f0a0a; border: 1px solid #7f1d1d; border-radius: 8px; padding: .75rem 1rem; font-size: .82rem; color: #fca5a5; margin-top: .75rem; display: none; }
  .footer { font-size: .7rem; color: #374151; text-align: center; margin-top: 1.5rem; }
  canvas { margin-top: 1rem; border-radius: 10px; }
</style>
</head>
<body>
<div class="container">
  <h1>Previsão de Consumo de Combustível <span class="tag">IA</span></h1>
  <p class="sub">TDE 3 · Regressão Linear Múltipla · Raciocínio Matemático para Computação</p>

  <div class="card">
    <p class="card-title">Características do veículo</p>
    <div class="grid">
      <div class="field">
        <label>Cilindrada (litros)</label>
        <input type="number" id="cil" placeholder="ex: 1.6" step="0.1" min="0.5" max="8">
        <p class="hint">Tamanho do motor — ex: 1.0, 1.6, 2.0</p>
      </div>
      <div class="field">
        <label>Peso (kg)</label>
        <input type="number" id="peso" placeholder="ex: 1200" step="10">
        <p class="hint">Peso total do veículo</p>
      </div>
      <div class="field">
        <label>Potência (cv)</label>
        <input type="number" id="cv" placeholder="ex: 120" step="1">
        <p class="hint">Cavalos de força do motor</p>
      </div>
      <div class="field">
        <label>Aceleração 0–100 km/h (s)</label>
        <input type="number" id="acel" placeholder="ex: 11.5" step="0.1">
        <p class="hint">Tempo em segundos</p>
      </div>
      <div class="field">
        <label>Ano do modelo</label>
        <input type="number" id="ano" placeholder="ex: 2022" step="1">
        <p class="hint">Ano de fabricação</p>
      </div>
    </div>
    <div class="error" id="err">Preencha todos os campos antes de prever.</div>
    <button class="predict" id="btn" onclick="prever()">Prever consumo →</button>
  </div>

  <div class="result-card" id="result">
    <p style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">Consumo estimado</p>
    <div class="result-num" id="rnum">—</div>
    <div class="result-unit">km por litro</div>
    <div id="rbadge"></div>

    <div class="metrics">
      <div class="metric"><div class="metric-val" id="ml100">—</div><div class="metric-lbl">Litros / 100km</div></div>
      <div class="metric"><div class="metric-val" id="mcusto">—</div><div class="metric-lbl">Custo / 100km (R$5,90)</div></div>
      <div class="metric"><div class="metric-val" id="mtank">—</div><div class="metric-lbl">Autonomia (50L)</div></div>
    </div>

    <div class="bar-wrap">
      <div class="bar-lbl"><span>Eficiência</span><span id="bpct">0%</span></div>
      <div class="bar-bg"><div class="bar-fill" id="bfill" style="width:0%"></div></div>
    </div>

    <div style="margin-top:1.5rem">
      <p style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">Comparação com a base de dados</p>
      <div style="position:relative;width:100%;height:220px"><canvas id="chart" role="img" aria-label="Gráfico de consumo previsto vs real">Gráfico comparativo de consumo.</canvas></div>
    </div>
  </div>

  <p class="footer">Modelo treinado com 15 veículos reais · R² = {{ r2 }} · RMSE ≈ {{ rmse }} km/L</p>
</div>

<script>
const BASE = [
  {n:'Fiat Uno',r:14.2},{n:'VW Gol',r:13.5},{n:'Fiat Mobi',r:14.8},
  {n:'HB20',r:13.0},{n:'Onix',r:12.8},{n:'Strada',r:11.5},
  {n:'VW Polo',r:11.8},{n:'Renegade',r:9.5},{n:'Civic',r:10.5},
  {n:'Corolla',r:10.0},{n:'Jetta',r:11.2},{n:'SW4',r:7.2},
  {n:'BMW M3',r:6.2},{n:'F-150',r:5.8},{n:'VW Up',r:12.5}
];

let chartInst = null;

async function prever() {
  const cil  = parseFloat(document.getElementById('cil').value);
  const peso = parseFloat(document.getElementById('peso').value);
  const cv   = parseFloat(document.getElementById('cv').value);
  const acel = parseFloat(document.getElementById('acel').value);
  const ano  = parseFloat(document.getElementById('ano').value);
  const err  = document.getElementById('err');
  const btn  = document.getElementById('btn');

  if ([cil, peso, cv, acel, ano].some(v => isNaN(v))) {
    err.style.display = 'block'; return;
  }
  err.style.display = 'none';
  btn.disabled = true;
  btn.textContent = 'Calculando...';

  const resp = await fetch('/prever', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cilindrada: cil, peso, potencia: cv, aceleracao: acel, ano })
  });
  const d = await resp.json();

  btn.disabled = false;
  btn.textContent = 'Prever consumo →';

  if (d.erro) { err.textContent = 'Erro: ' + d.erro; err.style.display = 'block'; return; }

  const val = d.consumo;
  document.getElementById('rnum').textContent   = val.toFixed(1);
  document.getElementById('ml100').textContent  = d.l100;
  document.getElementById('mcusto').textContent = 'R$' + d.custo.toFixed(2);
  document.getElementById('mtank').textContent  = d.autonomia + ' km';

  const pct = Math.min(100, Math.round((val / 16) * 100));
  document.getElementById('bpct').textContent = pct + '%';
  const fill = document.getElementById('bfill');
  fill.style.width = pct + '%';
  fill.style.background = val >= 13 ? '#10b981' : val >= 9 ? '#f59e0b' : '#ef4444';

  const badge = document.getElementById('rbadge');
  if (val >= 13)      badge.innerHTML = '<span class="badge badge-green">Excelente eficiência</span>';
  else if (val >= 10) badge.innerHTML = '<span class="badge badge-yellow">Boa eficiência</span>';
  else if (val >= 7)  badge.innerHTML = '<span class="badge badge-yellow">Eficiência regular</span>';
  else                badge.innerHTML = '<span class="badge badge-red">Baixa eficiência</span>';

  const res = document.getElementById('result');
  res.classList.add('show');

  const labels = [...BASE.map(b => b.n), 'Seu carro'];
  const reais  = [...BASE.map(b => b.r), null];
  const prevs  = [...BASE.map(_ => null), val];
  const bgPrev = labels.map((_, i) => i === labels.length - 1 ? '#6366f1' : 'transparent');

  if (chartInst) chartInst.destroy();
  chartInst = new Chart(document.getElementById('chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Base real', data: reais, backgroundColor: '#1e3a5f', borderColor: '#3b82f6', borderWidth: 1, borderRadius: 4 },
        { label: 'Seu carro', data: prevs, backgroundColor: bgPrev, borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#9ca3af', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#6b7280', font: { size: 9 }, maxRotation: 40 }, grid: { color: '#1f2937' } },
        y: { ticks: { color: '#6b7280' }, grid: { color: '#1f2937' }, title: { display: true, text: 'km/L', color: '#6b7280' }, min: 0, max: 18 }
      }
    }
  });
  res.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.querySelectorAll('input').forEach(i => i.addEventListener('keydown', e => { if (e.key === 'Enter') prever(); }));
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, r2=round(r2, 4), rmse=round(rmse, 4))

if __name__ == '__main__':
    import webbrowser, threading
    print("\n🚀 Abrindo no navegador em http://localhost:5000")
    threading.Timer(1.0, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(debug=False, port=5000)
