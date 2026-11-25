// Caelum Web Chat — Client Application (Zero-Dependency Vanilla JS)
const PERSONAS = {
  general: "You are Caelum, an advanced, highly intelligent, and friendly AI assistant running locally on an Intel Arc B580 GPU in pure Rust. Answer accurately, clearly, and concisely in the user's language.",
  rust: "You are a World-Class Principal Systems Engineer and Rust Expert. You specialize in zero-copy architectures, SIMD vectorization (AVX-VNNI, ARMv9 NEON), low-level concurrency, cache locality, and memory safety. Deliver battle-tested, idiomatic, high-performance Rust code with clear explanations.",
  cad: "You are the Lead CAD, Architectural & Structural Copilot for Lenout and Celestia. You specialize in parametric layout design, 3D spatial coordinate hierarchies, fabrication DXF generation, and strict JSON Schema action proposals.",
  hardware: "You are an Elite GPU & Hardware Architect specializing in the Intel Xe2-HPG Battlemage architecture (BMG G21), Vulkan 1.4, WGSL shaders, subgroups (16-32), and GGUF K-quantized matrix multiplication."
};

let sessions = JSON.parse(localStorage.getItem('caelum_sessions') || '[]');
let currentSessionId = localStorage.getItem('caelum_active_session') || null;
let abortController = null;

const defaultBase = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
let settings = JSON.parse(localStorage.getItem('caelum_settings') || JSON.stringify({
  endpoint: `${defaultBase}/v1/chat/completions`,
  temperature: 0.7,
  maxTokens: 512,
  persona: 'general',
  customSystem: ''
}));

const messagesContainer = document.getElementById('messages-container');
const welcomeHero = document.getElementById('welcome-hero');
const userPromptInput = document.getElementById('user-prompt');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const chatHistoryList = document.getElementById('chat-history-list');
const personaSelect = document.getElementById('persona-select');
const activePersonaTag = document.getElementById('active-persona-tag');

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderMarkdown(raw) {
  if (!raw) return '';
  let text = escapeHtml(raw);

  // Code blocks ```lang ... ```
  text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_match, lang, code) => {
    const language = lang || 'code';
    const unescapedCode = code
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&amp;/g, '&');
    return `<div class="code-block">
      <div class="code-block-header">
        <span>${escapeHtml(language)}</span>
        <button type="button" class="copy-code-btn" data-code="${encodeURIComponent(unescapedCode)}">Copy</button>
      </div>
      <pre><code>${code.trim()}</code></pre>
    </div>`;
  });

  // Inline code `code`
  text = text.replace(/`([^`]+)`/g, '<span class="inline-code">$1</span>');

  // Bold & Italic
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  const lines = text.split('\n');
  let out = '';
  let listTag = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      if (listTag !== 'ul') {
        if (listTag) out += `</${listTag}>`;
        out += '<ul>';
        listTag = 'ul';
      }
      out += `<li>${trimmed.substring(2)}</li>`;
    } else if (/^\d+\.\s/.test(trimmed)) {
      if (listTag !== 'ol') {
        if (listTag) out += `</${listTag}>`;
        out += '<ol>';
        listTag = 'ol';
      }
      out += `<li>${trimmed.replace(/^\d+\.\s/, '')}</li>`;
    } else {
      if (listTag) {
        out += `</${listTag}>`;
        listTag = null;
      }
      if (trimmed.length > 0 && !trimmed.startsWith('<div class="code-block"')) {
        out += `<p>${line}</p>`;
      } else {
        out += line;
      }
    }
  }
  if (listTag) out += `</${listTag}>`;
  return out;
}

function attachCopyHandlers() {
  document.querySelectorAll('.copy-code-btn').forEach(btn => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = 'true';
    btn.addEventListener('click', () => {
      const rawCode = decodeURIComponent(btn.getAttribute('data-code') || '');
      navigator.clipboard.writeText(rawCode).then(() => {
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove('copied');
        }, 1800);
      });
    });
  });
}

function initSessions() {
  if (sessions.length === 0) {
    createNewSession();
  } else {
    if (!currentSessionId || !sessions.find(s => s.id === currentSessionId)) {
      currentSessionId = sessions[0].id;
    }
    renderHistoryList();
    loadSessionMessages(currentSessionId);
  }
  if (settings.persona && PERSONAS[settings.persona]) {
    personaSelect.value = settings.persona;
    activePersonaTag.textContent = `Persona: ${personaSelect.options[personaSelect.selectedIndex].text}`;
  }
}

function createNewSession() {
  const newId = 'session_' + Date.now();
  const newSession = {
    id: newId,
    title: 'New Conversation',
    createdAt: Date.now(),
    messages: []
  };
  sessions.unshift(newSession);
  currentSessionId = newId;
  saveSessions();
  renderHistoryList();
  loadSessionMessages(newId);
  userPromptInput.focus();
}

function saveSessions() {
  localStorage.setItem('caelum_sessions', JSON.stringify(sessions));
  localStorage.setItem('caelum_active_session', currentSessionId);
}

function getActiveSession() {
  return sessions.find(s => s.id === currentSessionId);
}

function renderHistoryList() {
  chatHistoryList.innerHTML = '';
  const groupTitle = document.createElement('div');
  groupTitle.className = 'history-group-title';
  groupTitle.textContent = 'Recent Chats';
  chatHistoryList.appendChild(groupTitle);

  sessions.forEach(sess => {
    const item = document.createElement('div');
    item.className = `history-item ${sess.id === currentSessionId ? 'active' : ''}`;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'history-item-title';
    titleSpan.textContent = sess.title || 'Conversation';
    titleSpan.addEventListener('click', () => {
      currentSessionId = sess.id;
      saveSessions();
      renderHistoryList();
      loadSessionMessages(sess.id);
    });

    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'history-item-del';
    delBtn.title = 'Delete conversation';
    delBtn.textContent = '✕';
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(sess.id);
    });

    item.append(titleSpan, delBtn);
    chatHistoryList.appendChild(item);
  });
}

function deleteSession(id) {
  sessions = sessions.filter(s => s.id !== id);
  if (sessions.length === 0) {
    createNewSession();
  } else {
    if (currentSessionId === id) {
      currentSessionId = sessions[0].id;
    }
    saveSessions();
    renderHistoryList();
    loadSessionMessages(currentSessionId);
  }
}

function loadSessionMessages() {
  messagesContainer.innerHTML = '';
  const session = getActiveSession();
  if (!session || session.messages.length === 0) {
    messagesContainer.appendChild(welcomeHero);
    welcomeHero.style.display = 'flex';
    return;
  }
  welcomeHero.style.display = 'none';
  session.messages.forEach(msg => {
    appendMessageRow(msg.role, msg.content, msg.hud, false);
  });
  attachCopyHandlers();
  scrollToBottom();
}

function appendMessageRow(role, content, hud = null, isTyping = false) {
  welcomeHero.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'message-row';
  const isUser = role === 'user';

  row.innerHTML = `
    <div class="avatar ${isUser ? 'user' : 'assistant'}">
      ${isUser ? '👤' : '✦'}
    </div>
    <div class="message-body">
      <div class="message-sender">${isUser ? 'You' : 'Caelum Assistant'}</div>
      <div class="message-content ${isTyping ? 'typing-cursor' : ''}">${renderMarkdown(content)}</div>
      ${hud ? `<div class="message-hud">${hud}</div>` : ''}
    </div>
  `;
  messagesContainer.appendChild(row);
  attachCopyHandlers();
  scrollToBottom();
  return row;
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function handleSend() {
  const text = userPromptInput.value.trim();
  if (!text || abortController) return;

  const session = getActiveSession();
  if (!session) return;

  if (session.messages.length === 0) {
    session.title = text.slice(0, 32) + (text.length > 32 ? '...' : '');
    renderHistoryList();
  }

  session.messages.push({ role: 'user', content: text });
  appendMessageRow('user', text);
  userPromptInput.value = '';
  userPromptInput.style.height = 'auto';

  const assistantRow = appendMessageRow('assistant', '', null, true);
  const contentEl = assistantRow.querySelector('.message-content');
  const bodyEl = assistantRow.querySelector('.message-body');

  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';

  abortController = new AbortController();
  let fullResponse = '';
  let tokenCount = 0;
  let cachedTokens = 0;
  let firstTokenTime = null;
  const startTime = performance.now();

  const systemPrompt = (settings.customSystem || '').trim() || PERSONAS[settings.persona] || PERSONAS.general;
  const apiMessages = [
    { role: 'system', content: systemPrompt },
    ...session.messages.map(m => ({ role: m.role, content: m.content }))
  ];

  try {
    const res = await fetch(settings.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: apiMessages,
        temperature: parseFloat(settings.temperature),
        max_tokens: parseInt(settings.maxTokens, 10),
        stream: true
      }),
      signal: abortController.signal
    });

    if (!res.ok) {
      const err = await res.text();
      contentEl.classList.remove('typing-cursor');
      contentEl.innerHTML = `<span style="color:#ef4444;">Error ${res.status}: ${escapeHtml(err)}</span>`;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

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
            if (parsed.usage?.prompt_tokens_details?.cached_tokens != null) {
              cachedTokens = parsed.usage.prompt_tokens_details.cached_tokens;
            }
            const delta = parsed.choices?.[0]?.delta?.content || '';
            if (delta) {
              if (!firstTokenTime) firstTokenTime = performance.now();
              tokenCount++;
              fullResponse += delta;
              contentEl.innerHTML = renderMarkdown(fullResponse);
              scrollToBottom();
            }
          } catch (_) {}
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      contentEl.innerHTML += `<br><span style="color:#ef4444;">[Connection Error: ${escapeHtml(err.message)}]</span>`;
    }
  } finally {
    contentEl.classList.remove('typing-cursor');
    attachCopyHandlers();
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';

    const totalSec = Math.max((performance.now() - startTime) / 1000, 0.001);
    const ttft = firstTokenTime ? ((firstTokenTime - startTime) / 1000).toFixed(2) : '0.00';
    const tokPerSec = (tokenCount / totalSec).toFixed(1);
    const cacheNote = cachedTokens > 0 ? ` · ${cachedTokens} cached` : '';
    const hudStr = `<span class="hud-tag">${tokPerSec} tok/s</span> · ${tokenCount} tokens · TTFT ${ttft}s${cacheNote}`;

    const hudEl = document.createElement('div');
    hudEl.className = 'message-hud';
    hudEl.innerHTML = hudStr;
    bodyEl.appendChild(hudEl);

    session.messages.push({ role: 'assistant', content: fullResponse, hud: hudStr });
    saveSessions();
    abortController = null;
  }
}

sendBtn.addEventListener('click', handleSend);
stopBtn.addEventListener('click', () => {
  if (abortController) abortController.abort();
});

userPromptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

userPromptInput.addEventListener('input', () => {
  userPromptInput.style.height = 'auto';
  userPromptInput.style.height = Math.min(userPromptInput.scrollHeight, 180) + 'px';
});

document.querySelectorAll('.starter-card').forEach(card => {
  card.addEventListener('click', () => {
    userPromptInput.value = card.getAttribute('data-prompt') || '';
    userPromptInput.style.height = 'auto';
    handleSend();
  });
});

document.getElementById('new-chat-btn').addEventListener('click', createNewSession);

const sidebar = document.getElementById('sidebar');
document.getElementById('open-sidebar-btn').addEventListener('click', () => sidebar.classList.add('open'));
document.getElementById('close-sidebar-btn').addEventListener('click', () => sidebar.classList.remove('open'));

personaSelect.addEventListener('change', (e) => {
  settings.persona = e.target.value;
  localStorage.setItem('caelum_settings', JSON.stringify(settings));
  activePersonaTag.textContent = `Persona: ${personaSelect.options[personaSelect.selectedIndex].text}`;
});

// Dark/Light Theme Toggle (with prefers-color-scheme change listener)
const themeBtn = document.getElementById('theme-toggle-btn');
const themeIcon = document.getElementById('theme-icon');
const colorSchemeMq = window.matchMedia('(prefers-color-scheme: dark)');
let currentTheme = localStorage.getItem('caelum-theme') || (colorSchemeMq.matches ? 'dark' : 'light');

function applyTheme(theme, persist = false) {
  currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
  if (persist) localStorage.setItem('caelum-theme', theme);
}
applyTheme(currentTheme);
colorSchemeMq.addEventListener('change', (e) => {
  if (!localStorage.getItem('caelum-theme')) applyTheme(e.matches ? 'dark' : 'light');
});
themeBtn.addEventListener('click', () => applyTheme(currentTheme === 'dark' ? 'light' : 'dark', true));

// Settings Native <dialog>
const settingsDialog = document.getElementById('settings-dialog');
const openSettingsBtn = document.getElementById('open-settings-btn');
const closeSettingsX = document.getElementById('close-settings-x');
const settingsForm = document.getElementById('settings-form');
const valTemp = document.getElementById('val-temp');
const valMaxTokens = document.getElementById('val-max-tokens');
const settingTemp = document.getElementById('setting-temp');
const settingMaxTokens = document.getElementById('setting-max-tokens');
const settingEndpoint = document.getElementById('setting-endpoint');
const settingCustomSystem = document.getElementById('setting-custom-system');

openSettingsBtn.addEventListener('click', () => {
  settingEndpoint.value = settings.endpoint;
  settingTemp.value = settings.temperature;
  valTemp.textContent = settings.temperature;
  settingMaxTokens.value = settings.maxTokens;
  valMaxTokens.textContent = settings.maxTokens;
  settingCustomSystem.value = settings.customSystem || '';
  settingsDialog.showModal();
});

closeSettingsX.addEventListener('click', () => settingsDialog.close());
settingTemp.addEventListener('input', (e) => { valTemp.textContent = e.target.value; });
settingMaxTokens.addEventListener('input', (e) => { valMaxTokens.textContent = e.target.value; });

settingsForm.addEventListener('submit', () => {
  settings.endpoint = settingEndpoint.value.trim() || `${defaultBase}/v1/chat/completions`;
  settings.temperature = parseFloat(settingTemp.value);
  settings.maxTokens = parseInt(settingMaxTokens.value, 10);
  settings.customSystem = settingCustomSystem.value;
  localStorage.setItem('caelum_settings', JSON.stringify(settings));
});

document.getElementById('clear-history-btn').addEventListener('click', () => {
  if (confirm('ยืนยันที่จะลบประวัติการสนทนาทั้งหมดใช่หรือไม่?')) {
    sessions = [];
    localStorage.removeItem('caelum_sessions');
    createNewSession();
    settingsDialog.close();
  }
});

async function checkHealth() {
  const dot = document.getElementById('topbar-status-dot');
  const name = document.getElementById('topbar-model-name');
  const slotsPill = document.getElementById('topbar-slots-pill');
  const slotsText = document.getElementById('topbar-slots-text');
  const verBadge = document.getElementById('sidebar-version-badge');
  const footerHint = document.getElementById('engine-footer-hint');
  try {
    const res = await fetch(`${defaultBase}/health`, { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      const data = await res.json();
      dot.classList.add('online');
      name.textContent = `${data.model} (${(data.architecture || '').toUpperCase()} · ${data.kv_mode})`;
      if (data.version && verBadge) {
        verBadge.textContent = `v${data.version}`;
      }
      if (slotsPill && slotsText && typeof data.slots_total === 'number') {
        const active = data.slots_active ?? 0;
        const queued = data.queue_depth ?? 0;
        slotsPill.hidden = false;
        slotsPill.classList.toggle('busy', active > 0 || queued > 0);
        slotsText.textContent = queued > 0
          ? `Slots ${active}/${data.slots_total} · Q:${queued}`
          : `Slots ${active}/${data.slots_total}`;
      }
      if (footerHint && data.version) {
        footerHint.textContent = `Caelum Engine v${data.version} · ${data.batching === 'continuous' ? `Continuous Batching (${data.slots_total ?? 1} slots)` : 'Single-Slot'} · ${data.kv_mode} · 100% Private Local Inference`;
      }
      return;
    }
  } catch (_) {}
  dot.classList.remove('online');
  if (slotsPill) slotsPill.hidden = true;
  name.textContent = 'Caelum Offline (run caelum serve --gpu)';
}

initSessions();
checkHealth();
setInterval(checkHealth, 3000);
