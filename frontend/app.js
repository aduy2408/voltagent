const form = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const imageInput = document.querySelector('#image');
const imagePreview = document.querySelector('#image-preview');
const messages = document.querySelector('#messages');
const sessionInput = document.querySelector('#session');
const sessionList = document.querySelector('#session-list');
const sessionCount = document.querySelector('#session-count');
const sessionSearch = document.querySelector('#session-search');
const newSessionButton = document.querySelector('#new-session');
let sessions = [];
const savedSession = localStorage.getItem('voltagent-session');
if (savedSession) sessionInput.value = savedSession;

loadSessions().then(() => {
  if (savedSession && sessionList.querySelector(`[data-session-id="${savedSession}"]`)) {
    openSession(savedSession);
  }
});

newSessionButton.addEventListener('click', startNewSession);
sessionSearch.addEventListener('input', () => renderSessions(sessionSearch.value));
imageInput.addEventListener('change', () => renderImagePreview(imageInput.files[0]));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  const file = imageInput.files[0];
  if (!text && !file) return;
  addUserMessage(text || 'Inspect this image', file);
  prompt.value = '';
  const loading = add('VoltAgent', 'Thinking…', 'assistant', true);
  try {
    let imageId = null;
    if (file) {
      const formData = new FormData();
      formData.append('file', file);
      const upload = await fetch('/vision/upload', {method: 'POST', body: formData});
      const uploadPayload = await upload.json();
      if (!upload.ok) throw new Error(uploadPayload.detail || 'Image upload failed');
      imageId = uploadPayload.image_id;
      imageInput.value = '';
      clearImagePreview(false);
    }
    let response = await chatRequest(text || 'Inspect this image', imageId, sessionInput.value || null);
    let payload = await response.json();
    if (response.status === 404 && payload.detail === 'Session not found' && sessionInput.value) {
      sessionInput.value = '';
      localStorage.removeItem('voltagent-session');
      response = await chatRequest(text || 'Inspect this image', imageId, null);
      payload = await response.json();
    }
    if (!response.ok) throw new Error(payload.detail || 'Request failed');
    sessionInput.value = payload.session_id;
    localStorage.setItem('voltagent-session', payload.session_id);
    await loadSessions(payload.session_id);
    renderFormatted(loading.querySelector('.formatted'), payload.answer);
  } catch (error) {
    loading.querySelector('.formatted').textContent = `Error: ${error.message}`;
    loading.classList.add('error');
  }
});

function chatRequest(message, imageId, sessionId) {
  return fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message, image_id: imageId, session_id: sessionId}),
  });
}

async function loadSessions(selectedId = sessionInput.value) {
  const response = await fetch('/api/sessions');
  if (!response.ok) return;
  sessions = await response.json();
  renderSessions(sessionSearch.value, selectedId);
}

function renderSessions(filter = '', selectedId = sessionInput.value) {
  const query = filter.trim().toLowerCase();
  sessionList.replaceChildren();
  sessionCount.textContent = sessions.length;
  const visibleSessions = sessions.filter((session) => session.title.toLowerCase().includes(query));
  for (const session of visibleSessions) {
    const row = document.createElement('div');
    row.className = 'session-row';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'session-card';
    button.dataset.sessionId = session.id;
    button.innerHTML = `<strong>${escapeHtml(session.title)}</strong><small>${session.message_count} ${session.message_count === 1 ? 'message' : 'messages'} · ${formatDate(session.created_at)}</small>`;
    button.addEventListener('click', () => openSession(session.id));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'session-delete';
    remove.title = 'Delete conversation';
    remove.setAttribute('aria-label', `Delete ${session.title}`);
    remove.textContent = '×';
    remove.addEventListener('click', (event) => deleteSession(session.id, event));
    row.append(button, remove);
    sessionList.append(row);
  }
  if (selectedId) {
    const active = sessionList.querySelector(`[data-session-id="${CSS.escape(selectedId)}"]`);
    if (active) active.classList.add('active');
  }
}

async function deleteSession(sessionId, event) {
  event.stopPropagation();
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {method: 'DELETE'});
  if (!response.ok) return;
  if (sessionInput.value === sessionId) startNewSession();
  await loadSessions();
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
}

async function openSession(sessionId) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    startNewSession();
    return;
  }
  const session = await response.json();
  sessionInput.value = session.id;
  localStorage.setItem('voltagent-session', session.id);
  renderSessions(sessionSearch.value, session.id);
  messages.replaceChildren();
  for (const message of session.messages) {
    add(message.role === 'assistant' ? 'VoltAgent' : 'You', message.content, message.role === 'assistant' ? 'assistant' : 'user', message.role === 'assistant');
  }
  if (!session.messages.length) add('VoltAgent', 'Ready. Ask me to inspect a tool or route a task.', 'assistant', true);
}

function startNewSession() {
  sessionInput.value = '';
  localStorage.removeItem('voltagent-session');
  renderSessions(sessionSearch.value);
  messages.replaceChildren();
  add('VoltAgent', 'Ready. Ask me to inspect a tool or route a task.', 'assistant', true);
}

function add(name, text, role, formatted = false) {
  const item = document.createElement('div');
  item.className = `message ${role}`;
  const label = document.createElement('span');
  label.textContent = name;
  const body = document.createElement(formatted ? 'div' : 'p');
  body.className = formatted ? 'formatted' : '';
  if (formatted) renderFormatted(body, text);
  else body.textContent = text;
  item.append(label, body);
  messages.append(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}

function addUserMessage(text, file) {
  const item = add('You', text, 'user');
  if (file) {
    const image = document.createElement('img');
    image.className = 'message-image';
    image.src = URL.createObjectURL(file);
    image.alt = file.name;
    item.append(image);
  }
  return item;
}

function renderImagePreview(file) {
  clearImagePreview();
  if (!file) return;
  const image = document.createElement('img');
  image.className = 'preview-thumb';
  image.src = URL.createObjectURL(file);
  image.alt = file.name;
  const name = document.createElement('span');
  name.textContent = file.name;
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'preview-remove';
  remove.textContent = '×';
  remove.title = 'Remove image';
  remove.addEventListener('click', () => {
    imageInput.value = '';
    clearImagePreview();
  });
  imagePreview.append(image, name, remove);
  imagePreview.hidden = false;
}

function clearImagePreview(revoke = true) {
  if (revoke) {
    imagePreview.querySelectorAll('img').forEach((image) => URL.revokeObjectURL(image.src));
  }
  imagePreview.replaceChildren();
  imagePreview.hidden = true;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  })[character]);
}

function formatInline(value) {
  return value
    .replace(/\\\((.*?)\\\)/g, (_, math) => `<span class="math-inline">${simpleMathHtml(math)}</span>`)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}

function simpleMathHtml(source, display = false) {
  let value = escapeHtml(source).replace(/\\text\{([^{}]*)\}/g, '$1');
  value = value.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '<span class="frac"><span>$1</span><span>$2</span></span>');
  value = value.replace(/\\times/g, '×').replace(/\\Omega/g, 'Ω').replace(/\\pm/g, '±');
  value = value.replace(/\\([a-zA-Z]+)/g, '$1').replace(/\\/g, ' ');
  value = value.replace(/\^\{([^{}]*)\}/g, '<sup>$1</sup>').replace(/_\{([^{}]*)\}/g, '<sub>$1</sub>');
  value = value.replace(/[{}]/g, '');
  if (!display) return value;
  return value.split('\n').map((line) => `<div class="math-line">${line.trim()}</div>`).join('');
}

function renderFormatted(element, source) {
  element.innerHTML = renderMarkdown(source);
  element.querySelectorAll('.math-block').forEach((block) => {
    const latex = block.textContent.replace(/^\s*\\\[/, '').replace(/\\\]\s*$/, '').trim();
    if (window.katex) {
      window.katex.render(latex, block, {displayMode: true, throwOnError: false});
    } else {
      block.innerHTML = simpleMathHtml(latex, true);
    }
  });
  if (window.renderMathInElement) {
    window.renderMathInElement(element, {
      delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}],
      throwOnError: false,
    });
  }
  if (window.mermaid) {
    window.mermaid.initialize({startOnLoad: false, securityLevel: 'strict', theme: 'base', themeVariables: {
      background: '#111111', primaryColor: '#222222', primaryTextColor: '#eeeeee',
      primaryBorderColor: '#aaaaaa', lineColor: '#bbbbbb', secondaryColor: '#191919',
      tertiaryColor: '#111111', fontFamily: 'Inter, sans-serif',
    }});
    const diagrams = [...element.querySelectorAll('.mermaid')];
    if (diagrams.length) window.mermaid.run({nodes: diagrams});
  }
}

function renderMarkdown(source) {
  const lines = escapeHtml(String(source).replace(/\r\n/g, '\n')).split('\n');
  const output = [];
  let list = false;
  let code = false;
  let codeLines = [];
  let codeLanguage = '';
  let math = false;
  let mathLines = [];
  const closeList = () => {
    if (list) output.push('</ul>');
    list = false;
  };
  const closeCode = () => {
    if (code) output.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
    code = false;
    codeLines = [];
    codeLanguage = '';
  };
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim() === '\\[') {
      closeList();
      math = true;
      mathLines = [];
      continue;
    }
    if (math) {
      if (line.trim() === '\\]') {
        output.push(`<div class="math-block">\\[${mathLines.join('\n')}\\]</div>`);
        math = false;
        mathLines = [];
      } else {
        mathLines.push(line);
      }
      continue;
    }
    if (line.trim().startsWith('```')) {
      if (code) {
        if (codeLanguage === 'mermaid') {
          output.push(`<div class="mermaid">${codeLines.join('\n')}</div>`);
          code = false;
          codeLines = [];
          codeLanguage = '';
        } else closeCode();
      } else {
        closeList();
        code = true;
        codeLanguage = line.trim().slice(3).trim().toLowerCase();
      }
      continue;
    }
    if (code) { codeLines.push(line); continue; }
    if (!line.trim()) { closeList(); continue; }
    if (isTableSeparator(lines[index + 1])) {
      closeList();
      const header = tableCells(line);
      const rows = [];
      index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      index -= 1;
      output.push(`<table><thead><tr>${header.map((cell) => `<th>${formatInline(cell)}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${header.map((_, cellIndex) => `<td>${formatInline(row[cellIndex] || '')}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
      continue;
    }
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) {
      closeList();
      const level = line.indexOf(' ');
      output.push(`<h${level}>${formatInline(heading[1])}</h${level}>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      if (!list) { output.push('<ul>'); list = true; }
      output.push(`<li>${formatInline(bullet[1])}</li>`);
      continue;
    }
    closeList();
    output.push(`<p>${formatInline(line)}</p>`);
  }
  if (code) closeCode();
  closeList();
  return output.join('');
}

function tableCells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function isTableSeparator(line) {
  return Boolean(line && /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line));
}
