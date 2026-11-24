// Caelum Documentation & Live API Playground Script
const tablist = document.querySelector('[role="tablist"]');
const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

function selectTab(targetTab) {
  tabs.forEach(tab => {
    const isSelected = tab === targetTab;
    tab.setAttribute('aria-selected', isSelected ? 'true' : 'false');
    tab.toggleAttribute('focusgroupstart', isSelected);
  });
  panels.forEach(panel => {
    const match = panel.getAttribute('aria-labelledby') === targetTab.id;
    panel.hidden = !match;
  });
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (e) => {
    let index = tabs.indexOf(tab);
    if (e.key === 'ArrowRight') {
      index = (index + 1) % tabs.length;
      tabs[index].focus();
      selectTab(tabs[index]);
    } else if (e.key === 'ArrowLeft') {
      index = (index - 1 + tabs.length) % tabs.length;
      tabs[index].focus();
      selectTab(tabs[index]);
    }
  });
});

// Dark/Light Theme Toggle (with prefers-color-scheme change listener)
const themeBtn = document.getElementById('theme-btn');
const themeIcon = document.getElementById('theme-icon');
const themeText = document.getElementById('theme-text');
const colorSchemeMq = window.matchMedia('(prefers-color-scheme: dark)');
let currentTheme = localStorage.getItem('caelum-theme') || (colorSchemeMq.matches ? 'dark' : 'light');

function applyTheme(theme, persist = false) {
  currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
  themeText.textContent = theme === 'dark' ? 'Light' : 'Dark';
  if (persist) localStorage.setItem('caelum-theme', theme);
}
applyTheme(currentTheme);
colorSchemeMq.addEventListener('change', (e) => {
  if (!localStorage.getItem('caelum-theme')) applyTheme(e.matches ? 'dark' : 'light');
});
themeBtn.addEventListener('click', () => applyTheme(currentTheme === 'dark' ? 'light' : 'dark', true));

// Copy Buttons
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const code = btn.getAttribute('data-code');
    if (code) {
      navigator.clipboard.writeText(code).then(() => {
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove('copied');
        }, 1800);
      });
    }
  });
});

// Live Server Health Check
const defaultBase = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : window.location.origin;

async function checkServerHealth() {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const res = await fetch(`${defaultBase}/health`, { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      const data = await res.json();
      dot.classList.add('online');
      text.textContent = `Online: ${data.model} (${data.kv_mode})`;
      return;
    }
  } catch (_) {}
  dot.classList.remove('online');
  text.textContent = 'Offline (Run caelum serve)';
}
checkServerHealth();
setInterval(checkServerHealth, 5000);

// Live Interactive Playground
const pgEndpoint = document.getElementById('pg-endpoint');
if (window.location.protocol !== 'file:') {
  pgEndpoint.value = `${window.location.origin}/v1/chat/completions`;
}
const pgSendBtn = document.getElementById('pg-send-btn');
const pgOutputBox = document.getElementById('pg-output-box');
const pgStats = document.getElementById('pg-stats');

pgSendBtn.addEventListener('click', async () => {
  const endpoint = pgEndpoint.value.trim();
  const prompt = document.getElementById('pg-prompt').value.trim();
  const temp = parseFloat(document.getElementById('pg-temp').value) || 0.7;
  const maxTokens = parseInt(document.getElementById('pg-max-tokens').value, 10) || 256;
  if (!prompt) return;

  pgSendBtn.disabled = true;
  pgOutputBox.textContent = 'Connecting to Caelum Engine...\n';
  pgStats.textContent = '';
  const startTime = performance.now();
  let tokenCount = 0;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: prompt }],
        temperature: temp,
        max_tokens: maxTokens,
        stream: true
      })
    });

    if (!res.ok) {
      const errText = await res.text();
      pgOutputBox.textContent = `Error ${res.status}: ${errText}`;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    pgOutputBox.textContent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data: ')) {
          const payload = trimmed.slice(6);
          if (payload === '[DONE]') break;
          try {
            const parsed = JSON.parse(payload);
            const delta = parsed.choices?.[0]?.delta?.content || '';
            if (delta) {
              tokenCount++;
              pgOutputBox.textContent += delta;
              pgOutputBox.scrollTop = pgOutputBox.scrollHeight;
            }
          } catch (_) {}
        }
      }
    }
    const elapsedSec = Math.max((performance.now() - startTime) / 1000, 0.001);
    const tokPerSec = (tokenCount / elapsedSec).toFixed(1);
    pgStats.textContent = `Generated ${tokenCount} tokens in ${elapsedSec.toFixed(2)}s (${tokPerSec} tok/s)`;
  } catch (err) {
    pgOutputBox.textContent = `Connection failed: ${err.message}\nMake sure caelum serve is running with --gpu!`;
  } finally {
    pgSendBtn.disabled = false;
  }
});
