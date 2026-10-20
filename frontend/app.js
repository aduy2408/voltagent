const form = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const messages = document.querySelector('#messages');
const sessionInput = document.querySelector('#session');
const savedSession = localStorage.getItem('voltagent-session');
if (savedSession) sessionInput.value = savedSession;

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!text) return;
  add('You', text, 'user');
  prompt.value = '';
  const loading = add('VoltAgent', 'Thinking…', 'assistant');
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, session_id: sessionInput.value || null}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Request failed');
    sessionInput.value = payload.session_id;
    localStorage.setItem('voltagent-session', payload.session_id);
    loading.querySelector('p').textContent = payload.answer;
  } catch (error) {
    loading.querySelector('p').textContent = `Error: ${error.message}`;
    loading.classList.add('error');
  }
});

function add(name, text, role) {
  const item = document.createElement('div');
  item.className = `message ${role}`;
  const label = document.createElement('span');
  label.textContent = name;
  const body = document.createElement('p');
  body.textContent = text;
  item.append(label, body);
  messages.append(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}
