/* ============================================================
   data.js — Shared data, utilities, and backend integration
   ============================================================ */

'use strict';

/* ---- Mock transaction data (fallback / demo) ------------------------------ */
const MOCK_TRANSACTIONS = [
  { id:'TXN-00841', account:'ACC-1042', date:'2026-03-18', time:'09:14', type:'Cash Withdrawal',    amount:50000,  channel:'Counter',   branch:'BR-01', score:94, flag:1, risk:'high',   anomaly:'Rounded Amount, High Frequency' },
  { id:'TXN-00842', account:'ACC-1042', date:'2026-03-18', time:'09:22', type:'Cash Withdrawal',    amount:50000,  channel:'Counter',   branch:'BR-01', score:91, flag:1, risk:'high',   anomaly:'Rounded Amount, Near Duplicate' },
  { id:'TXN-00193', account:'ACC-2287', date:'2026-03-15', time:'14:05', type:'Account Transfer',   amount:187400, channel:'NEFT',      branch:'BR-02', score:88, flag:1, risk:'high',   anomaly:'Unusually Large Transaction' },
  { id:'TXN-00477', account:'ACC-3301', date:'2026-03-12', time:'11:30', type:'Internal Adjustment',amount:12000,  channel:'Internal',  branch:'BR-01', score:85, flag:1, risk:'high',   anomaly:'Reversal Abuse' },
  { id:'TXN-00512', account:'ACC-0091', date:'2026-03-10', time:'08:45', type:'Cash Withdrawal',    amount:200000, channel:'Counter',   branch:'BR-03', score:83, flag:1, risk:'high',   anomaly:'Dormant Account Activity' },
  { id:'TXN-00611', account:'ACC-4410', date:'2026-03-09', time:'16:58', type:'Account Transfer',   amount:10000,  channel:'IMPS',      branch:'BR-02', score:79, flag:1, risk:'high',   anomaly:'Rapid Fund Transfer' },
  { id:'TXN-00612', account:'ACC-4410', date:'2026-03-09', time:'17:01', type:'Account Transfer',   amount:10000,  channel:'IMPS',      branch:'BR-02', score:77, flag:1, risk:'high',   anomaly:'Rapid Fund Transfer' },
  { id:'TXN-00613', account:'ACC-4410', date:'2026-03-09', time:'17:04', type:'Account Transfer',   amount:10000,  channel:'IMPS',      branch:'BR-02', score:76, flag:1, risk:'high',   anomaly:'Rapid Fund Transfer' },
  { id:'TXN-00720', account:'ACC-1887', date:'2026-03-08', time:'10:20', type:'Reversal',           amount:45000,  channel:'Internal',  branch:'BR-01', score:72, flag:1, risk:'medium', anomaly:'Reversal Abuse' },
  { id:'TXN-00731', account:'ACC-1887', date:'2026-03-08', time:'10:35', type:'Reversal',           amount:45000,  channel:'Internal',  branch:'BR-01', score:70, flag:1, risk:'medium', anomaly:'Near Duplicate, Reversal Abuse' },
  { id:'TXN-00310', account:'ACC-5502', date:'2026-03-07', time:'13:10', type:'Loan Disbursement',  amount:500000, channel:'Counter',   branch:'BR-03', score:68, flag:1, risk:'medium', anomaly:'Unusually Large Transaction' },
  { id:'TXN-00419', account:'ACC-2100', date:'2026-03-06', time:'15:45', type:'Vendor Payment',     amount:25000,  channel:'NEFT',      branch:'BR-02', score:65, flag:1, risk:'medium', anomaly:'Repeated Vendor Payment' },
  { id:'TXN-00420', account:'ACC-2100', date:'2026-03-06', time:'15:52', type:'Vendor Payment',     amount:25000,  channel:'NEFT',      branch:'BR-02', score:64, flag:1, risk:'medium', anomaly:'Repeated Vendor Payment' },
  { id:'TXN-00421', account:'ACC-2100', date:'2026-03-06', time:'16:01', type:'Vendor Payment',     amount:25000,  channel:'NEFT',      branch:'BR-02', score:63, flag:1, risk:'medium', anomaly:'Repeated Vendor Payment' },
  { id:'TXN-00880', account:'ACC-3310', date:'2026-03-05', time:'12:00', type:'Cash Deposit',       amount:9900,   channel:'Counter',   branch:'BR-01', score:58, flag:1, risk:'medium', anomaly:'Split Transaction' },
  { id:'TXN-00881', account:'ACC-3310', date:'2026-03-05', time:'12:08', type:'Cash Deposit',       amount:9900,   channel:'Counter',   branch:'BR-01', score:57, flag:1, risk:'medium', anomaly:'Split Transaction' },
  { id:'TXN-00882', account:'ACC-3310', date:'2026-03-05', time:'12:15', type:'Cash Deposit',       amount:9900,   channel:'Counter',   branch:'BR-01', score:55, flag:1, risk:'medium', anomaly:'Split Transaction' },
  { id:'TXN-00215', account:'ACC-7711', date:'2026-03-04', time:'09:00', type:'EMI Repayment',      amount:3200,   channel:'Online',    branch:'BR-02', score:42, flag:1, risk:'low',    anomaly:'Sudden Behaviour Change' },
  { id:'TXN-00301', account:'ACC-6604', date:'2026-03-03', time:'11:22', type:'Cash Withdrawal',    amount:30000,  channel:'Counter',   branch:'BR-03', score:38, flag:1, risk:'low',    anomaly:'Rounded Amount' },
  { id:'TXN-00155', account:'ACC-1120', date:'2026-03-01', time:'10:05', type:'Account Transfer',   amount:5500,   channel:'NEFT',      branch:'BR-01', score:31, flag:1, risk:'low',    anomaly:'Unusual Frequency' },
  // Clean transactions
  { id:'TXN-00100', account:'ACC-1042', date:'2026-03-01', time:'10:00', type:'Cash Deposit',       amount:12000,  channel:'Counter',   branch:'BR-01', score:8,  flag:0, risk:'low',    anomaly:'' },
  { id:'TXN-00101', account:'ACC-1042', date:'2026-03-05', time:'09:30', type:'EMI Repayment',      amount:4500,   channel:'Online',    branch:'BR-01', score:5,  flag:0, risk:'low',    anomaly:'' },
  { id:'TXN-00102', account:'ACC-1042', date:'2026-03-10', time:'14:20', type:'Cash Withdrawal',    amount:3000,   channel:'Counter',   branch:'BR-01', score:11, flag:0, risk:'low',    anomaly:'' },
  { id:'TXN-00200', account:'ACC-4410', date:'2026-03-01', time:'11:00', type:'Cash Deposit',       amount:8000,   channel:'Counter',   branch:'BR-02', score:6,  flag:0, risk:'low',    anomaly:'' },
  { id:'TXN-00201', account:'ACC-4410', date:'2026-03-05', time:'15:00', type:'Account Transfer',   amount:2000,   channel:'NEFT',      branch:'BR-02', score:9,  flag:0, risk:'low',    anomaly:'' },
];

/* ---- Anomaly type summary ------------------------------------------------- */
const ANOMALY_TYPES = [
  { name:'Rounded Amount',             count:4, risk:'medium', desc:'Transactions involving suspiciously round figures such as Rs.10,000 or Rs.50,000, repeated within a short window.' },
  { name:'Near Duplicate',             count:3, risk:'high',   desc:'Same or very similar payments recorded more than once within minutes, differing only slightly in amount or timestamp.' },
  { name:'Unusually Large Transaction',count:2, risk:'high',   desc:"Single transaction amounts that are significantly higher than the account's historical average." },
  { name:'Rapid Fund Transfer',        count:3, risk:'high',   desc:'Quick movement of funds across multiple accounts in sequence, each individual hop appearing routine.' },
  { name:'Reversal Abuse',             count:3, risk:'high',   desc:'Excessive reversal or correction entries not corresponding to clear legitimate causes.' },
  { name:'Dormant Account Activity',   count:1, risk:'high',   desc:'Significant transactions appearing in accounts that had been inactive for an extended period.' },
  { name:'Split Transaction',          count:3, risk:'medium', desc:'A large payment broken into several smaller amounts, often to stay below a reporting threshold.' },
  { name:'Repeated Vendor Payment',    count:3, risk:'medium', desc:'Multiple payments to the same recipient in quick succession with no clear business justification.' },
  { name:'Sudden Behaviour Change',    count:1, risk:'low',    desc:'A sharp shift in how an account is being used, with no clear explanation from transaction history.' },
  { name:'High Frequency',             count:1, risk:'high',   desc:'Excessive number of transactions from a single account within a short duration.' },
  { name:'Unusual Frequency',          count:1, risk:'low',    desc:'Transaction frequency slightly above normal for this account type and period.' },
];

/* ---- Run summary ---------------------------------------------------------- */
const RUN_SUMMARY = {
  file:          'transactions_march_2026.csv',
  processed:     new Date('2026-03-22T14:32:00'),
  total:         4217,
  flagged:       20,
  highRisk:      8,
  mediumRisk:    9,
  lowRisk:       3,
  contamination: 0.05,
  model:         'IsolationForest (n_estimators=100)',
  processingMs:  3840,
};

/* ============================================================
   BACKEND INTEGRATION
   ============================================================ */

// Flask serves both frontend and backend on the same origin,
// so we use a relative URL — works locally and after deployment with zero changes.
const BACKEND_URL = '';

/**
 * POST a CSV/Excel file to the backend and return JSON.
 * Includes a 60-second timeout and one automatic retry on network failure.
 */
async function fetchBackendData(file, attempt = 1) {
  const formData = new FormData();
  formData.append('file', file);

  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), 180000); // 3 min timeout for large files

  try {
    const response = await fetch(`${BACKEND_URL}/upload`, {
      method: 'POST',
      body:   formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error ${response.status}: ${text}`);
    }
    return response.json();

  } catch (err) {
    clearTimeout(timeoutId);

    const isNetworkErr = err.name === 'AbortError' ||
                         err.message.toLowerCase().includes('fetch') ||
                         err.message.toLowerCase().includes('network') ||
                         err.message.toLowerCase().includes('failed');

    if (isNetworkErr && attempt === 1) {
      console.warn('Fetch attempt 1 failed — retrying in 2s:', err.message);
      await new Promise(r => setTimeout(r, 2000));
      return fetchBackendData(file, 2);
    }
    if (err.name === 'AbortError') {
      throw new Error('Request timed out (>60s). Is the Flask server running on port 5000?');
    }
    throw err;
  }
}

/**
 * Map a single raw backend row to the UI transaction shape.
 * Handles flexible/uppercase column names from the Flask pipeline.
 */
function mapBackendRow(row) {
  // Score: backend stores 0-1 float; UI expects 0-100 integer
  const rawScore = row.Score ?? row.score ?? row.anomaly_score ?? 0;
  const score = rawScore <= 1 ? Math.round(rawScore * 100) : Math.round(Number(rawScore));

  // Flag
  const flag = (row.Anomaly ?? row.flag ?? row.anomaly_flag ?? 0) ? 1 : 0;

  // Risk string: "High Risk" / "Medium Risk" / "Low Risk" → "high" / "medium" / "low"
  const rawRisk = String(row.Risk ?? row.risk ?? row.risk_level ?? '').toLowerCase();
  let risk = 'low';
  if (rawRisk.includes('high'))   risk = 'high';
  else if (rawRisk.includes('med')) risk = 'medium';

  // Date + time: split combined datetime if needed
  let date = String(row.date ?? row.DATE ?? '');
  let time = String(row.time ?? row.TIME ?? '');
  if (!time && date.includes('T')) {
    [date, time] = date.split('T');
    time = time.slice(0, 5);
  } else if (!time && date.includes(' ')) {
    const parts = date.split(' ');
    date = parts[0];
    time = parts[1] ? parts[1].slice(0, 5) : '';
  }

  // Resolve account — routes.py normalises to 'account_id' but catch extras here too
  const acctRaw = row.account_id ?? row.ACCOUNT_ID ?? row.account ?? row['Account No'] ??
                  row['Account No.'] ?? row.Acct_ID ?? row.acct_id ?? row.AccountID ?? null;
  const account = acctRaw !== null ? String(acctRaw) : 'ACC-?';

  // Resolve anomaly type — backend now always sets anomaly_type
  const anomaly = String(row.anomaly_type ?? row.anomaly ?? row.ANOMALY ?? row.AnomalyType ?? '');

  return {
    id:      String(row.transaction_id ?? row.TRANSACTION_ID ?? row.id ?? row.Trans_ID ?? 'TXN-?'),
    account,
    date,
    time,
    type:    String(row.transaction_type ?? row.type ?? row.TYPE ?? row.Trans_Type ?? ''),
    amount:  parseFloat(row.amount ?? row.Amount ?? row.AMOUNT ?? 0),
    channel: String(row.channel ?? row.Channel ?? row.CHANNEL ?? ''),
    branch:  String(row.branch_code ?? row.branch ?? row.BRANCH ?? row.Branch_Code ?? ''),
    score,
    flag,
    risk,
    anomaly,
  };
}

function mapBackendToUI(rows) {
  return Array.isArray(rows) ? rows.map(mapBackendRow) : [];
}

/**
 * Primary data accessor used by every page.
 * Returns mapped real data if a backend run was stored, else mock data.
 */
function getTransactions() {
  try {
    const stored = localStorage.getItem('transactions');
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return mapBackendToUI(parsed);
      }
    }
  } catch (_) {
    // corrupt storage — fall through to mock
  }
  return MOCK_TRANSACTIONS;
}

/**
 * Build a live RUN_SUMMARY from a backend response + mapped transactions.
 */
function buildRunSummary(result, mapped) {
  const flagged = mapped.filter(t => t.flag === 1);
  return {
    file:          result.filename ?? 'uploaded_file.csv',
    processed:     new Date(),
    total:         result.total_records ?? mapped.length,
    flagged:       result.fraud_count   ?? flagged.length,
    highRisk:      flagged.filter(t => t.risk === 'high').length,
    mediumRisk:    flagged.filter(t => t.risk === 'medium').length,
    lowRisk:       flagged.filter(t => t.risk === 'low').length,
    contamination: 0.05,
    model:         'IsolationForest (n_estimators=100)',
    processingMs:  result.processing_ms ?? 0,
  };
}

/* ============================================================
   SHARED UTILITIES
   ============================================================ */

const Utils = {
  fmt: {
    currency: n => 'Rs.' + Number(n).toLocaleString('en-IN'),
    date:     s => new Date(s).toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'numeric' }),
    pct:      n => (n * 100).toFixed(1) + '%',
    score:    n => n.toString().padStart(2, '0'),
  },

  scoreColor(score) {
    if (score >= 75) return '#ff6b6b';
    if (score >= 50) return '#e0ac44';
    return '#52c491';
  },

  riskClass(risk) {
    return { high:'high', medium:'medium', low:'low' }[risk] || 'low';
  },

  sidebar(active) {
    const txns    = getTransactions();
    const flagCnt = txns.filter(t => t.flag === 1).length || RUN_SUMMARY.flagged;

    const links = [
      { key:'upload',    href:'index.html',    icon:'&#8679;', label:'Upload &amp; Process' },
      { key:'dashboard', href:'dashboard.html',icon:'&#9672;', label:'Dashboard' },
      { key:'flagged',   href:'flagged.html',  icon:'&#9873;', label:'Flagged Transactions', badge: flagCnt },
      { key:'account',   href:'account.html',  icon:'&#9678;', label:'Account Detail' },
      { key:'explorer',  href:'explorer.html', icon:'&#8862;', label:'Anomaly Explorer' },
      { key:'export',    href:'export.html',   icon:'&#8595;', label:'Export Report' },
    ];
    return `
      <div class="sidebar-section">Navigation</div>
      ${links.map(l => `
        <a href="${l.href}" class="nav-link ${l.key === active ? 'active' : ''}">
          <span class="icon">${l.icon}</span>
          ${l.label}
          ${l.badge ? `<span class="badge">${l.badge}</span>` : ''}
        </a>
      `).join('')}
    `;
  },

  topbar(status = 'ready') {
    const dot   = status === 'ready'
      ? '<span class="dot"></span>'
      : '<span class="dot idle"></span>';
    const label = status === 'ready' ? 'Results loaded' : 'No file processed';
    return `
      <a href="index.html" class="topbar-logo">MOMENTUM<span> / audit</span></a>
      <span class="topbar-sep"></span>
      <span class="topbar-status">${dot} ${label}</span>
    `;
  },

  toast(msg, type = 'success') {
    let t = document.getElementById('toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'toast';
      t.className = 'toast';
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.className = `toast ${type === 'error' ? 'error' : ''} show`;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => t.classList.remove('show'), 3200);
  },

  sortBy(arr, key, asc = true) {
    return [...arr].sort((a, b) => {
      if (a[key] < b[key]) return asc ? -1 : 1;
      if (a[key] > b[key]) return asc ? 1 : -1;
      return 0;
    });
  },

  paginate(arr, page, perPage) {
    const start = (page - 1) * perPage;
    return arr.slice(start, start + perPage);
  },
};
