'use strict';

// ── State (in-memory only — nothing is persisted, per the no-history rule) ──
// { id, role:'user'|'bot'|'bot-pending', text, safetyLevel, suggestedQuestions,
//   matchedTopic, geneMetadata, llmUsed, fallbackUsed, feedbackState, isWelcome }
let messages = [];
let isSending = false;
let lastTopic = null;

const CONTEXT_WINDOW = 6;

// ── Demo questions (shown in the demo strip) ─────────────────────────────────
const DEMO_QUESTIONS = [
  { label: 'מה זה VUS?',            question: 'מה זה VUS?' },
  { label: 'VUS vs. pathogenic',    question: 'מה ההבדל בין VUS לבין ממצא pathogenic?' },
  { label: 'האם VUS משתנה?',        question: 'האם VUS יכול להשתנות בעתיד?' },
  { label: 'VUS והחלטות רפואיות',   question: 'למה בדרך כלל לא מקבלים החלטות רפואיות רק לפי VUS?' },
  { label: 'מה זה גן?',             question: 'מה זה גן?' },
];

// ── Feedback preset reasons ───────────────────────────────────────────────
const FEEDBACK_REASONS = [
  'התשובה לא הייתה רלוונטית',
  'התשובה לא הייתה ברורה',
  'התשובה לא ענתה על שאלתי',
  'מידע חסר',
];

const PENDING_TEXT_HE = 'כותב תשובה...';

// ── Welcome message ──────────────────────────────────────────────────────────
const WELCOME_TEXT = [
  'שלום! אני כאן לעזור להבין מושגים גנטיים כלליים לאחר שפגשתם יועץ גנטי.',
  '',
  'אני יכול/ה להסביר: VUS, נשאות, דפוסי תורשה, מה ידוע על גן לפי ClinVar, ומה כדאי לשאול בפגישה הבאה.',
  '',
  'אינני מפרש/ת תוצאות בדיקה אישיות, לא מאבחן/ת, ולא מחליפ/ה ייעוץ גנטי.',
  '',
  '⚠ אין להזין שם, תעודת זהות, טלפון, אימייל, או מספר ממצא מבדיקה אישית.',
].join('\n');

const WELCOME_CHIPS = [
  'מה זה VUS?',
  'מה זה נשאות?',
  'מה ידוע על BRCA1?',
  'יש לי VUS ב-BRCA2 — מה זה אומר?',
  'מה לשאול את הגנטיקאי?',
  'מה זה pathogenic?',
];

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Privacy notice — dismiss on button click
  const privacyOk = byId('privacy-ok');
  if (privacyOk) {
    privacyOk.addEventListener('click', () => {
      const notice = byId('privacy-notice');
      if (notice) notice.remove();
      byId('chat-input').focus();
    });
  }

  byId('btn-clear-chat').addEventListener('click', clearConversation);
  byId('chat-form').addEventListener('submit', onSend);
  byId('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(e); }
  });
  byId('demo-strip-close').addEventListener('click', () => {
    byId('demo-strip').hidden = true;
  });
  byId('error-banner-close').addEventListener('click', hideErrorBanner);

  renderDemoStrip();
  injectWelcomeMessage();
});

function byId(id) { return document.getElementById(id); }

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ── Error banner ──────────────────────────────────────────────────────────────
function showErrorBanner(msg) {
  const banner = byId('error-banner');
  byId('error-banner-text').textContent = msg;
  banner.hidden = false;
}

function hideErrorBanner() {
  byId('error-banner').hidden = true;
}

// ── Welcome message ──────────────────────────────────────────────────────────
function injectWelcomeMessage() {
  addMessage('bot', WELCOME_TEXT, 'general_information', WELCOME_CHIPS, null, null, false, false, true);
}

// ── Demo strip ────────────────────────────────────────────────────────────────
function renderDemoStrip() {
  const scroll = byId('demo-scroll');
  scroll.innerHTML = '';
  const label = document.createElement('span');
  label.className = 'demo-strip-label';
  label.textContent = 'שאלות לדוגמה:';
  scroll.appendChild(label);
  DEMO_QUESTIONS.forEach(({ label: btnLabel, question }) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'demo-btn';
    btn.textContent = btnLabel;
    btn.title = question;
    btn.addEventListener('click', () => {
      if (isSending) return;
      sendQuestion(question, null);
    });
    scroll.appendChild(btn);
  });
}

// ── Clear conversation ────────────────────────────────────────────────────────
function clearConversation() {
  messages = [];
  lastTopic = null;
  byId('demo-strip').hidden = false;
  hideErrorBanner();
  renderMessages();
  injectWelcomeMessage();
}

// ── Conversation context payload ──────────────────────────────────────────────
function buildConversationContext() {
  return messages
    .filter((m) => (m.role === 'user' || m.role === 'bot') && !m.isWelcome)
    .slice(-CONTEXT_WINDOW)
    .map((m) => ({
      role: m.role === 'bot' ? 'assistant' : 'user',
      content: m.text,
      matched_topic: m.role === 'bot' ? (m.matchedTopic || null) : null,
    }));
}

// ── Send ──────────────────────────────────────────────────────────────────────
function onSend(e) {
  e.preventDefault();
  if (isSending) return;
  const input = byId('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  hideErrorBanner();
  sendQuestion(text, null);
}

async function sendQuestion(question, topic) {
  if (isSending) return;
  isSending = true;
  setSendingUiState(true);

  const conversationContext = buildConversationContext();
  const lastTopicForThisTurn = lastTopic;

  addMessage('user', question);
  const pendingId = addMessage('bot-pending', PENDING_TEXT_HE);

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        topic: topic || undefined,
        conversation_context: conversationContext.length ? conversationContext : undefined,
        last_topic: lastTopicForThisTurn || undefined,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'שגיאה בשרת.' }));
      const errText = err.detail || 'אירעה שגיאה בשרת. אנא נסה שוב.';
      replaceMessage(pendingId, 'bot', errText, 'out_of_scope', [], null, null, false, true);
      showErrorBanner(`שגיאת שרת (${resp.status}): ${errText}`);
      return;
    }

    const data = await resp.json();
    replaceMessage(
      pendingId, 'bot',
      data.answer, data.safety_level,
      data.suggested_questions || [],
      data.matched_topic || null,
      data.gene_metadata || null,
      data.llm_used || false,
      data.fallback_used !== undefined ? data.fallback_used : true,
    );
    // If the backend already included the draft, pre-populate it (no second request needed)
    if (data.unverified_gene_draft) {
      const botMsg = messages.find((m) => m.id === pendingId);
      if (botMsg) {
        botMsg.unverifiedDraft = data.unverified_gene_draft;
        botMsg.unverifiedDraftState = 'loaded';
      }
    }
    // General education AI draft -- auto-populated, shown immediately
    if (data.unverified_general_draft) {
      const botMsg = messages.find((m) => m.id === pendingId);
      if (botMsg) {
        botMsg.generalDraft = data.unverified_general_draft;
      }
    }
    lastTopic = data.matched_topic || lastTopic;
  } catch (err) {
    const msg = 'לא ניתן היה להתחבר לשרת. בדקי את החיבור לאינטרנט ונסי שוב.';
    replaceMessage(pendingId, 'bot', msg, 'out_of_scope', [], null, null, false, true);
    showErrorBanner(msg);
  } finally {
    isSending = false;
    setSendingUiState(false);
  }
}

function setSendingUiState(sending) {
  byId('btn-send').disabled = sending;
  document.querySelectorAll('.suggested-chip').forEach((c) => { c.disabled = sending; });
  document.querySelectorAll('.demo-btn').forEach((b) => { b.disabled = sending; });
  document.querySelectorAll('.feedback-btn').forEach((b) => { b.disabled = sending; });
}

// ── Message state ─────────────────────────────────────────────────────────────
let _nextId = 1;

function addMessage(role, text, safetyLevel, suggestedQuestions, matchedTopic, geneMetadata, llmUsed, fallbackUsed, isWelcome) {
  const id = _nextId++;
  messages.push({
    id, role, text, safetyLevel,
    suggestedQuestions: suggestedQuestions || [],
    matchedTopic: matchedTopic || null,
    geneMetadata: geneMetadata || null,
    llmUsed: llmUsed || false,
    fallbackUsed: fallbackUsed !== undefined ? fallbackUsed : true,
    feedbackState: null,
    isWelcome: isWelcome || false,
    unverifiedDraft: null,
    unverifiedDraftState: null,
    generalDraft: null,
  });
  renderMessages();
  return id;
}

function replaceMessage(id, role, text, safetyLevel, suggestedQuestions, matchedTopic, geneMetadata, llmUsed, fallbackUsed) {
  const msg = messages.find((m) => m.id === id);
  if (msg) {
    msg.role = role; msg.text = text; msg.safetyLevel = safetyLevel;
    msg.suggestedQuestions = suggestedQuestions || [];
    msg.matchedTopic = matchedTopic || null;
    msg.geneMetadata = geneMetadata || null;
    msg.llmUsed = llmUsed || false;
    msg.fallbackUsed = fallbackUsed !== undefined ? fallbackUsed : true;
    if (msg.feedbackState === undefined) msg.feedbackState = null;
    if (msg.unverifiedDraft === undefined) msg.unverifiedDraft = null;
    if (msg.unverifiedDraftState === undefined) msg.unverifiedDraftState = null;
    if (msg.generalDraft === undefined) msg.generalDraft = null;
  }
  renderMessages();
}

// ── Feedback ──────────────────────────────────────────────────────────────────

async function submitFeedback(msgId, helpful, reason) {
  const msg = messages.find((m) => m.id === msgId);
  if (!msg) return;
  msg.feedbackState = 'submitted';
  renderMessages();

  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        helpful,
        reason: reason || null,
        matched_topic: msg.matchedTopic || null,
        safety_level: msg.safetyLevel || null,
        question_length: (() => {
          const prev = messages.find((m) => m.id === msgId - 1);
          return prev ? (prev.text || '').length : null;
        })(),
      }),
    });
  } catch (_) {
    // Feedback submission failure is silent — never disrupts the user session
  }
}

// ── Gene metadata expandable panel ────────────────────────────────────────────
function buildGeneMetadataHtml(meta) {
  if (!meta) return '';
  const total = (meta.total_variants != null)
    ? Number(meta.total_variants).toLocaleString('he-IL') : '—';
  const tierLabel = {
    tier1: 'מידע מסוכם ומאושר על הגן',
    tier2: 'נתונים סטטיסטיים ממאגר ClinVar (ללא סיכום ביולוגי מאושר)',
    tier3: 'הגן אינו כלול במאגר המקומי',
  };
  const rows = [
    ['גן',                  meta.gene_symbol || '—'],
    ['מקור',                meta.data_source || 'ClinVar (NCBI)'],
    ['רשומות במאגר',        total],
    ['סוג המידע',            tierLabel[meta.answer_tier] || '—'],
    ['ניסוח',               meta.llm_used ? 'פתיחה בסיוע בינה מלאכותית + תוכן מאושר' : 'תוכן מאושר בלבד'],
    ['נמצא במאגר ClinVar',  meta.found_in_index ? 'כן' : 'לא'],
  ];
  const tableRows = rows
    .map(([k, v]) => `<tr><th>${escHtml(k)}</th><td>${escHtml(String(v))}</td></tr>`)
    .join('');

  // Significance breakdown and phenotypes are shown in the separate ClinVar
  // technical card for Tier 2 — do not duplicate them here.
  let sigHtml = '';
  if (meta.answer_tier !== 'tier2' && meta.significance_breakdown && Object.keys(meta.significance_breakdown).length) {
    const sigRows = Object.entries(meta.significance_breakdown)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([k, v]) => `<tr><th>${escHtml(k)}</th><td>${Number(v).toLocaleString('he-IL')}</td></tr>`)
      .join('');
    sigHtml = `<p class="gene-meta-section-label">סיווגים קליניים:</p><table class="gene-meta-table">${sigRows}</table>`;
  }

  let phenoHtml = '';
  if (meta.answer_tier !== 'tier2' && meta.top_phenotypes && meta.top_phenotypes.length) {
    const items = meta.top_phenotypes
      .slice(0, 6)
      .map(p => `<li>${escHtml(p)}</li>`)
      .join('');
    phenoHtml = `<p class="gene-meta-section-label">מצבים קשורים מדווחים:</p><ul class="gene-meta-phenotypes">${items}</ul>`;
  }

  return (
    `<details class="gene-meta-panel">` +
    `<summary class="gene-meta-summary"><span class="gene-meta-indicator">&#9654;</span>מקור המידע ופרטים נוספים</summary>` +
    `<table class="gene-meta-table">${tableRows}</table>` +
    sigHtml + phenoHtml +
    `</details>`
  );
}

// ── Rendering ─────────────────────────────────────────────────────────────────
const _AI_BADGE_HE = 'מידע AI לא מאומת — להסבר כללי בלבד.';

function safetyBadgeClass(level) {
  return { contains_identifying_info: 'msg-warning', requires_genetic_counselor: 'msg-notice', out_of_scope: 'msg-notice', general_information: '' }[level] || '';
}

function renderMessages() {
  const container = byId('chat-messages');
  container.innerHTML = '';

  const lastBotIndex = messages.reduce((acc, m, idx) => (m.role === 'bot' ? idx : acc), -1);

  messages.forEach((m, idx) => {
    const row = document.createElement('div');
    row.className = `msg-row msg-row--${m.role === 'user' ? 'user' : 'bot'}`;

    const bubble = document.createElement('div');
    const isBot = m.role !== 'user';
    bubble.className = `msg-bubble msg-bubble--${m.role === 'user' ? 'user' : 'bot'} ${isBot ? safetyBadgeClass(m.safetyLevel) : ''}`;

    let html = `<p class="msg-text">${escHtml(m.text)}</p>`;

    if (isBot && m.role !== 'bot-pending' && !m.isWelcome) {
      if (m.geneMetadata) {
        html += buildGeneMetadataHtml(m.geneMetadata);
      } else if (m.llmUsed) {
        html += `<p class="msg-llm-badge">✦ ניסוח בינה מלאכותית</p>`;
      }
    }

    bubble.innerHTML = html;

    // Suggested question chips (last bot message only, including welcome)
    if (isBot && m.role !== 'bot-pending') {
      if (idx === lastBotIndex && m.suggestedQuestions && m.suggestedQuestions.length) {
        const chipsWrap = document.createElement('div');
        chipsWrap.className = 'suggested-chips';
        m.suggestedQuestions.forEach((q) => {
          const chip = document.createElement('button');
          chip.type = 'button'; chip.className = 'suggested-chip';
          chip.textContent = q; chip.disabled = isSending;
          chip.addEventListener('click', () => { if (!isSending) sendQuestion(q, null); });
          chipsWrap.appendChild(chip);
        });
        bubble.appendChild(chipsWrap);
      }

      // ClinVar technical card (Tier 2 only, skip for welcome/pending)
      if (!m.isWelcome) {
        const techCard = buildClinvarTechCard(m.geneMetadata);
        if (techCard) bubble.appendChild(techCard);
      }

      // Unverified draft opt-in card (Tier 2 genes only, skip for welcome)
      if (!m.isWelcome) {
        const draftCard = buildUnverifiedDraftCard(m);
        if (draftCard) bubble.appendChild(draftCard);
      }

      // General education AI draft card (auto-shown when present)
      if (!m.isWelcome) {
        const genCard = buildGeneralDraftCard(m);
        if (genCard) bubble.appendChild(genCard);
      }

      // Feedback row (skip for welcome message)
      if (!m.isWelcome) {
        const feedbackRow = buildFeedbackRow(m);
        if (feedbackRow) bubble.appendChild(feedbackRow);
      }
    }

    row.appendChild(bubble);
    container.appendChild(row);
  });

  container.scrollTop = container.scrollHeight;
}

// ── ClinVar technical details card (Tier 2 only) ─────────────────────────────

function buildClinvarTechCard(meta) {
  if (!meta || meta.answer_tier !== 'tier2') return null;
  const hasSig = meta.significance_breakdown && Object.keys(meta.significance_breakdown).length;
  const hasPheno = meta.top_phenotypes && meta.top_phenotypes.length;
  if (!hasSig && !hasPheno && meta.total_variants == null) return null;

  const card = document.createElement('div');
  card.className = 'clinvar-tech-card';

  const details = document.createElement('details');
  details.className = 'clinvar-tech-details';

  const summary = document.createElement('summary');
  summary.className = 'clinvar-tech-summary';
  summary.textContent = 'פרטים טכניים ממאגר ClinVar';
  details.appendChild(summary);

  const subtitle = document.createElement('p');
  subtitle.className = 'clinvar-tech-subtitle';
  subtitle.textContent = 'מידע זה אינו מפרש את התוצאה האישית שלך';
  details.appendChild(subtitle);

  if (meta.total_variants != null) {
    const total = document.createElement('p');
    total.className = 'clinvar-tech-total';
    total.textContent = `סך הכל רשומות במאגר: ${Number(meta.total_variants).toLocaleString('he-IL')}`;
    details.appendChild(total);
  }

  if (hasSig) {
    const lbl = document.createElement('p');
    lbl.className = 'clinvar-tech-label';
    lbl.textContent = 'סיווגים קליניים מדווחים:';
    details.appendChild(lbl);

    const table = document.createElement('table');
    table.className = 'clinvar-tech-table';
    Object.entries(meta.significance_breakdown)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .forEach(([k, v]) => {
        const row = document.createElement('tr');
        const th = document.createElement('th'); th.textContent = k;
        const td = document.createElement('td'); td.textContent = Number(v).toLocaleString('he-IL');
        row.appendChild(th); row.appendChild(td);
        table.appendChild(row);
      });
    details.appendChild(table);
  }

  if (hasPheno) {
    const lbl = document.createElement('p');
    lbl.className = 'clinvar-tech-label';
    lbl.textContent = 'מצבים מדווחים במאגר:';
    details.appendChild(lbl);

    const ul = document.createElement('ul');
    ul.className = 'clinvar-tech-phenotypes';
    meta.top_phenotypes.slice(0, 6).forEach(p => {
      const li = document.createElement('li');
      li.textContent = p;
      ul.appendChild(li);
    });
    details.appendChild(ul);
  }

  const note = document.createElement('p');
  note.className = 'clinvar-tech-note';
  note.textContent = 'הנתונים לעיל לקוחים ממאגר ClinVar בלבד ואינם מהווים פרשנות אישית של ממצא הבדיקה.';
  details.appendChild(note);

  card.appendChild(details);
  return card;
}

// ── Unverified gene draft opt-in card ────────────────────────────────────────

async function loadUnverifiedDraft(msgId) {
  const msg = messages.find((m) => m.id === msgId);
  if (!msg) return;
  msg.unverifiedDraftState = 'loading';
  renderMessages();

  const userMsg = messages.slice().reverse().find(
    (m) => m.role === 'user' && m.id < msgId
  );
  const question = userMsg ? userMsg.text : (msg.geneMetadata ? msg.geneMetadata.gene_symbol || '' : '');

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        last_topic: msg.matchedTopic || undefined,
        include_unverified_gene_draft: true,
      }),
    });
    if (resp.ok) {
      const data = await resp.json();
      msg.unverifiedDraft = data.unverified_gene_draft || null;
    } else {
      msg.unverifiedDraft = null;
    }
  } catch (_) {
    msg.unverifiedDraft = null;
  }
  msg.unverifiedDraftState = 'loaded';
  renderMessages();
}

function buildUnverifiedDraftCard(msg) {
  const meta = msg.geneMetadata;
  // Only show draft UI if the backend confirmed a draft was actually generated.
  // unverified_gene_draft_available is now set AFTER the draft attempt, so it
  // is true only when a real draft object exists.
  if (!meta || meta.answer_tier !== 'tier2' || !meta.unverified_gene_draft_available) return null;

  const card = document.createElement('div');
  card.className = 'unverified-draft-card';

  if (msg.unverifiedDraftState === null) {
    // Draft was not pre-populated in the initial response — offer manual fetch
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'unverified-draft-btn';
    btn.textContent = 'הצג/י מידע לא מאומת';
    btn.title = 'מידע שנוצר אוטומטית על ידי בינה מלאכותית — לא עבר בדיקה מקצועית';
    btn.addEventListener('click', () => loadUnverifiedDraft(msg.id));
    card.appendChild(btn);
  } else if (msg.unverifiedDraftState === 'loading') {
    const loading = document.createElement('span');
    loading.className = 'unverified-draft-loading';
    loading.textContent = 'טוען מידע ניסיוני...';
    card.appendChild(loading);
  } else if (msg.unverifiedDraftState === 'loaded') {
    if (msg.unverifiedDraft) {
      const d = msg.unverifiedDraft;
      const details = document.createElement('details');
      details.className = 'unverified-draft-details';
      details.open = true;

      const summary = document.createElement('summary');
      summary.className = 'unverified-draft-summary';
      summary.textContent = 'מידע ניסיוני על הגן';
      details.appendChild(summary);

      const badge = document.createElement('p');
      badge.className = 'ai-badge';
      badge.textContent = _AI_BADGE_HE;
      details.appendChild(badge);

      const text = document.createElement('p');
      text.className = 'unverified-draft-text';
      text.textContent = d.text_he || '';
      details.appendChild(text);

      card.appendChild(details);
    }
    // If loaded but no draft object: show nothing. The backend sets
    // unverified_gene_draft_available=false when draft fails, so in practice
    // this branch (loaded + null) should not be reached in normal flow.
    // Do NOT show any error message here — absence of draft is silent.
  }

  return card;
}

// ── General education AI draft badge ──────────────────────────────────
// Compact badge shown when backend returns unverified_general_draft.

function buildGeneralDraftCard(msg) {
  if (!msg.generalDraft) return null;
  const d = msg.generalDraft;
  const badge = document.createElement('p');
  badge.className = 'ai-badge';
  badge.textContent = _AI_BADGE_HE;
  return badge;
}

// ── Feedback row ──────────────────────────────────────────────────────────────

function buildFeedbackRow(msg) {
  if (msg.role === 'bot-pending') return null;

  const wrap = document.createElement('div');
  wrap.className = 'feedback-row';

  if (msg.feedbackState === 'submitted') {
    const thanks = document.createElement('span');
    thanks.className = 'feedback-thanks';
    thanks.textContent = 'תודה על המשוב!';
    wrap.appendChild(thanks);
    return wrap;
  }

  // "Was this helpful?" label
  const label = document.createElement('span');
  label.className = 'feedback-label';
  label.textContent = 'האם התשובה עזרה?';
  wrap.appendChild(label);

  // Helpful button
  const btnYes = document.createElement('button');
  btnYes.type = 'button';
  btnYes.className = `feedback-btn feedback-btn--yes ${msg.feedbackState === 'helpful' ? 'feedback-btn--active' : ''}`;
  btnYes.title = 'כן, עזר';
  btnYes.textContent = 'כן';
  btnYes.disabled = isSending;
  btnYes.addEventListener('click', () => {
    msg.feedbackState = 'helpful';
    renderMessages();
    submitFeedback(msg.id, true, null);
  });
  wrap.appendChild(btnYes);

  // Not helpful button — reveals reason selector
  const btnNo = document.createElement('button');
  btnNo.type = 'button';
  btnNo.className = `feedback-btn feedback-btn--no ${msg.feedbackState === 'not_helpful' ? 'feedback-btn--active' : ''}`;
  btnNo.title = 'לא עזר';
  btnNo.textContent = 'לא';
  btnNo.disabled = isSending;
  btnNo.addEventListener('click', () => {
    msg.feedbackState = 'not_helpful';
    renderMessages();
  });
  wrap.appendChild(btnNo);

  // Reason selector (only when "not helpful" clicked, before submission)
  if (msg.feedbackState === 'not_helpful') {
    const reasonWrap = document.createElement('div');
    reasonWrap.className = 'feedback-reason-wrap';

    const sel = document.createElement('select');
    sel.className = 'feedback-reason-select';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'סיבה (אופציונלי)';
    placeholder.disabled = true;
    placeholder.selected = true;
    sel.appendChild(placeholder);

    FEEDBACK_REASONS.forEach((r) => {
      const opt = document.createElement('option');
      opt.value = r; opt.textContent = r;
      sel.appendChild(opt);
    });

    const sendBtn = document.createElement('button');
    sendBtn.type = 'button';
    sendBtn.className = 'feedback-send-btn';
    sendBtn.textContent = 'שלח';
    sendBtn.addEventListener('click', () => {
      submitFeedback(msg.id, false, sel.value || null);
    });

    reasonWrap.appendChild(sel);
    reasonWrap.appendChild(sendBtn);
    wrap.appendChild(reasonWrap);
  }

  return wrap;
}
