// Chatbot NeoRAG: gửi câu hỏi tới /api/chat và vẽ câu trả lời theo luồng sự kiện SSE.
// Thứ tự sự kiện server gửi: status → (rewrite) → sources → token... → done | error.
"use strict";

const $ = (sel) => document.querySelector(sel);
const els = {
  messages: $("#messages"),
  welcome: $("#welcome"),
  welcomeExamples: $("#welcome-examples"),
  chips: $("#chips"),
  form: $("#form"),
  input: $("#input"),
  send: $("#send"),
  status: $("#status"),
  metaLine: $("#meta-line"),
  newChat: $("#new-chat"),
  clearChat: $("#clear-chat"),
  history: $("#history"),
  app: $("#app"),
  backdrop: $("#backdrop"),
  openSidebar: $("#open-sidebar"),
  openInsights: $("#open-insights"),
  insightsDot: $("#insights-dot"),
  banner: $("#banner"),
  charCount: $("#char-count"),
  sourcesBody: $("#sources-body"),
  sourcesCount: $("#sources-count"),
  insightsQuestion: $("#insights-question"),
  perfBody: $("#perf-body"),
  perfTotal: $("#perf-total"),
};

// Tên các bước theo thứ tự chạy, khớp khóa `timings` server gửi trong sự kiện done.
const TIMING_LABELS = {
  followup: "Hiểu ngữ cảnh",
  rewrite: "Viết lại câu hỏi",
  search: "Tìm kiếm",
  rerank: "Xếp hạng lại",
  generate: "Sinh câu trả lời",
};

const STEP_TEXT = {
  followup: "Đang hiểu ngữ cảnh câu hỏi…",
  retrieve: "Đang tìm tài liệu liên quan…",
  generate: "Đang viết câu trả lời…",
};
const INPUT_MAX_HEIGHT = 160; // khớp max-height của textarea trong CSS
const MAX_CHARS = 1000; // khớp max_length của ChatRequest.question ở server
const CHAR_WARN = 800; // từ mức này mới hiện bộ đếm, để bình thường ô nhập gọn
// Không nhận được sự kiện nào trong ngần này giây thì coi như server treo: dừng và báo lỗi,
// thay vì để người dùng nhìn vòng xoay mãi. Bình thường giữa hai sự kiện chỉ vài giây.
const IDLE_TIMEOUT_MS = 60_000;
const RECONNECT_MS = 3_000;

// Mốc trùng với CSS: dưới mốc này cột tương ứng thành ngăn kéo.
const DRAWER_MQ = {
  insights: matchMedia("(max-width: 1279px)"),
  sidebar: matchMedia("(max-width: 767px)"),
};

const state = {
  conversations: [], // mọi cuộc trò chuyện đã lưu, mới nhất trước
  current: null, // cuộc đang mở; null = cuộc mới chưa hỏi câu nào (chưa lưu)
  busy: false,
  controller: null, // AbortController của câu hỏi đang chạy (nút Dừng)
  inspected: null, // tin nhắn bot mà panel bên phải đang hiển thị nguồn/thời gian
};

// ---------------------------------------------------------------------------
// Tiện ích vẽ
// ---------------------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Markdown tối thiểu cho câu trả lời: thoát HTML trước (chống chèn mã), rồi mới đổi **đậm** và gạch đầu dòng.
// Xuống dòng giữ nguyên nhờ white-space: pre-wrap của .bubble.
function renderAnswer(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^[-*] /gm, "• ");
}

function logoAvatar() {
  const avatar = el("span", "avatar");
  avatar.setAttribute("aria-hidden", "true");
  avatar.innerHTML = '<svg><use href="#logo"/></svg>';
  return avatar;
}

// Chỉ tự cuộn xuống khi người dùng đang ở gần cuối: họ cuộn lên đọc lại thì không bị kéo xuống.
function nearBottom() {
  const m = els.messages;
  return m.scrollHeight - m.scrollTop - m.clientHeight < 80;
}
function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setStatus(stateName, text) {
  els.status.dataset.state = stateName;
  els.status.querySelector(".status-text").textContent = text;
}

function setBusy(busy) {
  state.busy = busy;
  els.send.classList.toggle("stop", busy);
  els.send.querySelector(".send-label").textContent = busy ? "Dừng" : "Gửi";
  els.send.setAttribute("aria-label", busy ? "Dừng trả lời" : "Gửi câu hỏi");
  updateSendButton();
  if (!busy) setStatus("idle", "Sẵn sàng");
}

function updateSendButton() {
  // Đang trả lời: nút luôn bấm được (để Dừng). Rảnh: chỉ bấm được khi đã gõ gì đó.
  els.send.disabled = !state.busy && !els.input.value.trim();
}

// ---------------------------------------------------------------------------
// Panel bên phải: nguồn tham khảo + hiệu năng
// Mỗi tin nhắn bot giữ dữ liệu của riêng nó (view.sources, view.done); panel vẽ lại
// từ dữ liệu đó, nên bấm vào câu trả lời cũ là xem lại được nguồn của câu đó.
// ---------------------------------------------------------------------------

function sourceCard(doc, index) {
  const card = el("article", "source");
  card.dataset.club = doc.label;
  card.dataset.index = index;
  const head = el("div", "source-head");
  const badge = el("span", "badge", doc.label);
  badge.dataset.club = doc.label;
  head.append(el("span", "source-num", String(index + 1)), badge, el("span", "source-topic", doc.topic));
  head.querySelector(".source-topic").title = doc.topic;

  const text = el("p", "source-snippet", doc.snippet);
  card.append(head, text);
  if (doc.text.length > doc.snippet.length) {
    // Mở/thu toàn văn ngay trong thẻ: không cần hộp thoại, không mất ngữ cảnh.
    const toggle = el("button", "link-btn", "Xem toàn văn");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      text.textContent = open ? doc.snippet : doc.text;
      text.classList.toggle("full", !open);
      toggle.textContent = open ? "Xem toàn văn" : "Thu gọn";
      toggle.setAttribute("aria-expanded", String(!open));
    });
    card.append(toggle);
  }
  return card;
}

function renderSources(view) {
  const body = els.sourcesBody;
  body.replaceChildren();
  const docs = view.sources ?? [];
  // Tài liệu không có đáp án: nguồn lấy về không liên quan, hiện ra chỉ gây hiểu nhầm.
  if (view.done?.no_info) {
    body.append(el("p", "empty-note", "Không có tài liệu nào chứa thông tin cho câu hỏi này."));
    els.sourcesCount.hidden = true;
    return;
  }
  if (!docs.length) {
    body.append(el("p", "empty-note", view.finished ? "Không có nguồn." : "Đang tìm tài liệu…"));
    els.sourcesCount.hidden = true;
    return;
  }
  docs.forEach((doc, i) => body.append(sourceCard(doc, i)));
  // Màn hẹp, ngăn kéo nguồn đang đóng: chấm hồng trên nút 📖 báo có nguồn mới để xem.
  // Chỉ khi đang trả lời thật (view chưa xong), không phải lúc mở lại cuộc cũ.
  if (!view.finished && DRAWER_MQ.insights.matches && !els.app.classList.contains("show-insights")) {
    els.insightsDot.hidden = false;
  }
  // Nguồn tới trước câu trả lời: trong lúc model còn viết, đánh dấu là "đang dùng".
  body.classList.toggle("pending", !view.done);
  els.sourcesCount.hidden = false;
  els.sourcesCount.textContent = String(docs.length);
}

function renderTimings(view) {
  const body = els.perfBody;
  body.replaceChildren();
  const done = view.done;
  if (!done) {
    body.append(el("p", "empty-note", view.finished ? "Câu trả lời chưa hoàn tất." : "Đang đo…"));
    els.perfTotal.hidden = true;
    return;
  }
  const list = el("ul", "timings");
  for (const [key, label] of Object.entries(TIMING_LABELS)) {
    if (!(key in done.timings)) continue; // bước không chạy (ví dụ câu hỏi không cần ngữ cảnh)
    const sec = done.timings[key];
    const bar = el("span", "bar");
    const fill = el("i");
    // Thanh dài theo tỉ lệ trên tổng thời gian; tối thiểu 2% để bước rất nhanh vẫn thấy được.
    fill.style.setProperty("--w", `${Math.max(2, (sec / done.total) * 100)}%`);
    bar.append(fill);
    const item = el("li");
    item.append(el("span", "", label), bar, el("b", "", `${sec.toFixed(1)}s`));
    list.append(item);
  }
  body.append(list);
  if (done.timings.first_piece !== undefined) {
    const first = el("div", "first-token", "Chữ đầu tiên sau ");
    first.append(el("b", "", `${done.timings.first_piece.toFixed(1)}s`));
    body.append(first);
  }
  els.perfTotal.hidden = false;
  els.perfTotal.textContent = `${done.total.toFixed(1)}s`;
}

function showInsights(view, highlightIndex) {
  if (state.inspected && state.inspected !== view) state.inspected.msg.classList.remove("inspected");
  state.inspected = view;
  view.msg.classList.add("inspected");
  els.insightsQuestion.hidden = false;
  els.insightsQuestion.textContent = view.question;
  renderSources(view);
  renderTimings(view);
  if (highlightIndex !== undefined) {
    const card = els.sourcesBody.querySelector(`.source[data-index="${highlightIndex}"]`);
    if (card) {
      card.classList.add("highlight");
      // Chỉ cuộn trong ô nguồn. scrollIntoView sẽ cuộn cả trang khi ngăn kéo còn nằm ngoài màn hình
      // (tablet/điện thoại), làm lệch toàn bộ bố cục.
      const box = els.sourcesBody;
      const top = card.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
      box.scrollTo({ top: top - 4, behavior: "smooth" });
    }
  }
}

function clearInsights() {
  state.inspected = null;
  els.insightsQuestion.hidden = true;
  els.sourcesCount.hidden = true;
  els.perfTotal.hidden = true;
  els.sourcesBody.replaceChildren(el("p", "empty-note", "Tài liệu dùng để trả lời sẽ hiện ở đây."));
  els.perfBody.replaceChildren(el("p", "empty-note", "Thời gian từng bước của câu trả lời."));
}

// Chip nguồn ngay dưới câu trả lời: "1 ProPTIT · Lịch sử…". Bấm để xem nguồn đó ở panel phải.
function sourceChips(view) {
  const row = el("div", "source-chips");
  row.append(el("span", "source-chips-label", "Nguồn:"));
  view.sources.forEach((doc, i) => {
    const chip = el("button", "source-chip");
    chip.type = "button";
    chip.dataset.club = doc.label;
    chip.title = `${doc.label} · ${doc.topic}`;
    chip.append(el("span", "source-chip-num", String(i + 1)), el("span", "source-chip-topic", doc.topic));
    chip.addEventListener("click", () => {
      if (DRAWER_MQ.insights.matches) openDrawer("insights"); // màn hẹp: panel đang ẩn, mở ra trước
      showInsights(view, i);
    });
    row.append(chip);
  });
  return row;
}

// ---------------------------------------------------------------------------
// Lưu cuộc trò chuyện trong trình duyệt (localStorage): server không giữ gì.
// Cuộc: { id, title, updatedAt, turns: [Turn] }
// Turn (một lượt hỏi-đáp): { question, answer, standalone, sources, done, error, stopped }
//   done = dữ liệu sự kiện "done" (no_info, timings, total); null nếu lượt chưa xong.
// ---------------------------------------------------------------------------

const STORE_KEY = "neorag.conversations.v1"; // đổi "v1" khi đổi cấu trúc dữ liệu
const CURRENT_KEY = "neorag.current.v1";
const MAX_CONVERSATIONS = 30;

// localStorage có thể bị chặn (chế độ riêng tư, chính sách trình duyệt): mọi lần đọc/ghi đều bọc try,
// hỏng thì chatbot vẫn chạy bình thường, chỉ là không nhớ được sau khi tải lại trang.
function loadStore() {
  try {
    const data = JSON.parse(localStorage.getItem(STORE_KEY));
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function saveStore() {
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  state.conversations.length = Math.min(state.conversations.length, MAX_CONVERSATIONS);
  while (true) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(state.conversations));
      localStorage.setItem(CURRENT_KEY, state.current?.id ?? "");
      return;
    } catch (err) {
      // Đầy bộ nhớ (~5MB): bỏ dần cuộc cũ nhất rồi ghi lại. Lỗi khác (bị chặn) thì thôi.
      if (err.name !== "QuotaExceededError" || state.conversations.length <= 1) return;
      state.conversations.pop();
    }
  }
}

function newId() {
  return crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

// Lịch sử gửi server để hiểu câu hỏi nối tiếp: chỉ các lượt trả lời trọn vẹn (bỏ lỗi, bỏ bị Dừng).
function historyForServer() {
  return (state.current?.turns ?? [])
    .filter((t) => t.done && !t.error && !t.stopped)
    .flatMap((t) => [
      { role: "user", content: t.question },
      { role: "assistant", content: t.answer },
    ]);
}

// ---------------------------------------------------------------------------
// Danh sách cuộc trò chuyện ở cột trái
// ---------------------------------------------------------------------------

function startOfDay(ts) {
  const d = new Date(ts);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function dayGroup(ts) {
  const days = Math.round((startOfDay(Date.now()) - startOfDay(ts)) / 86_400_000);
  if (days <= 0) return "Hôm nay";
  if (days === 1) return "Hôm qua";
  if (days < 7) return "7 ngày qua";
  return "Cũ hơn";
}

function relativeTime(ts) {
  const minutes = Math.floor((Date.now() - ts) / 60_000);
  if (minutes < 1) return "vừa xong";
  if (minutes < 60) return `${minutes} phút`;
  if (minutes < 24 * 60 && dayGroup(ts) === "Hôm nay") return `${Math.floor(minutes / 60)} giờ`;
  return new Date(ts).toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" });
}

function renderHistoryList() {
  const nav = els.history;
  nav.replaceChildren();
  if (!state.conversations.length) {
    nav.append(el("p", "empty-note", "Các cuộc trò chuyện của bạn sẽ hiện ở đây."));
    return;
  }
  let group = null;
  for (const conv of state.conversations) {
    const g = dayGroup(conv.updatedAt);
    if (g !== group) {
      group = g;
      nav.append(el("div", "history-group", g));
    }
    const item = el("div", "history-item");
    const active = conv === state.current;
    item.classList.toggle("active", active);

    const open = el("button", "history-open");
    open.type = "button";
    open.title = conv.title;
    if (active) open.setAttribute("aria-current", "true");
    open.append(el("span", "history-title", conv.title), el("span", "history-time", relativeTime(conv.updatedAt)));
    open.addEventListener("click", () => openConversation(conv));

    const del = el("button", "history-del");
    del.type = "button";
    del.setAttribute("aria-label", `Xóa cuộc trò chuyện: ${conv.title}`);
    del.innerHTML = '<svg class="icon" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>';
    del.addEventListener("click", () => deleteConversation(conv));

    item.append(open, del);
    nav.append(item);
  }
}

// ---------------------------------------------------------------------------
// Tin nhắn
// ---------------------------------------------------------------------------

function addUserMessage(text) {
  const msg = el("div", "msg user");
  msg.append(el("div", "bubble", text));
  els.messages.append(msg);
  return msg;
}

// Khung tin nhắn bot rỗng: phần thân và dòng trạng thái được điền dần khi sự kiện tới.
function addBotMessage(turn) {
  const msg = el("div", "msg bot");
  const body = el("div", "msg-body");
  const progress = el("div", "progress");
  progress.append(el("span", "spinner"), el("span", "progress-text", STEP_TEXT.retrieve));
  body.append(progress);
  msg.append(logoAvatar(), body);
  els.messages.append(msg);
  // view = phần giao diện của một lượt; view.sources/done dùng chung với turn để panel phải đọc.
  return { msg, body, progress, bubble: null, turn, question: turn.question, sources: turn.sources, done: null };
}

function ensureBubble(view) {
  if (!view.bubble) {
    view.bubble = el("div", "bubble");
    view.progress.before(view.bubble);
  }
  return view.bubble;
}

function addRewrite(view, standalone) {
  // Cho người dùng thấy chatbot đã hiểu câu hỏi nối tiếp thế nào.
  const line = el("div", "rewrite", "↺ Hiểu là: ");
  line.append(el("em", "", standalone));
  view.body.prepend(line);
}

function showError(view, message, detail) {
  view.finished = true;
  view.msg.className = "msg bot error";
  view.msg.querySelector(".avatar").textContent = "!";
  view.body.replaceChildren();
  const bubble = el("div", "bubble", message);
  const retry = el("button", "retry", "Thử lại");
  retry.type = "button";
  retry.addEventListener("click", () => {
    // Bỏ lượt lỗi khỏi cuộc trò chuyện rồi hỏi lại đúng câu đó (không thêm bong bóng câu hỏi mới).
    if (state.current) state.current.turns = state.current.turns.filter((t) => t !== view.turn);
    view.msg.remove();
    ask(view.question, { retry: true });
  });
  bubble.append(retry);
  if (detail) bubble.title = detail; // chi tiết kỹ thuật khi rê chuột, không làm rối giao diện
  view.body.append(bubble);
}

function showStopped(view, answer) {
  view.finished = true;
  view.progress.remove();
  if (answer) ensureBubble(view).innerHTML = renderAnswer(answer);
  view.body.append(el("div", "msg-meta", "Đã dừng"));
}

function finishAnswer(view, answer, done) {
  view.finished = true;
  view.done = done;
  ensureBubble(view);
  view.progress.remove();
  view.bubble.innerHTML = renderAnswer(answer);
  if (done.no_info) {
    view.msg.classList.add("no-info");
    view.bubble.prepend(el("span", "tag", "Ngoài phạm vi tài liệu"));
  } else if (view.sources.length) {
    view.body.append(sourceChips(view));
  }
  const meta = el("div", "msg-meta");
  meta.append(el("span", "", `${done.total.toFixed(1)}s`));
  const copy = el("button", "meta-btn", "Sao chép");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(answer);
      copy.textContent = "Đã chép ✓";
    } catch {
      copy.textContent = "Không chép được";
    }
    setTimeout(() => (copy.textContent = "Sao chép"), 1500);
  });
  meta.append(copy);
  view.body.append(meta);
}

// Vẽ lại một lượt đã lưu (khi mở cuộc trò chuyện cũ hoặc tải lại trang).
function renderSavedTurn(turn) {
  addUserMessage(turn.question);
  const view = addBotMessage(turn);
  if (turn.standalone) addRewrite(view, turn.standalone);
  if (turn.error) showError(view, turn.error.message, turn.error.detail);
  else if (turn.done) finishAnswer(view, turn.answer, turn.done);
  else showStopped(view, turn.answer); // bị Dừng, hoặc trang bị đóng giữa lúc đang trả lời
  return view;
}

// ---------------------------------------------------------------------------
// Đọc luồng SSE từ fetch (EventSource chỉ hỗ trợ GET nên tự tách sự kiện)
// ---------------------------------------------------------------------------

async function* readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replace(/\r\n/g, "\n");
    let end;
    // Mỗi sự kiện kết thúc bằng một dòng trống; một lần đọc có thể chứa nửa sự kiện hoặc nhiều sự kiện.
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

// ---------------------------------------------------------------------------
// Hỏi một câu
// ---------------------------------------------------------------------------

async function ask(question, { retry = false } = {}) {
  question = question.trim();
  if (!question || state.busy) return;

  // Lấy lịch sử trước khi thêm lượt mới (lượt mới chưa có câu trả lời).
  const history = historyForServer();
  // Câu đầu của cuộc mới: tạo cuộc ngay để cột trái hiện tên cuộc trò chuyện liền.
  if (!state.current) {
    state.current = { id: newId(), title: question.slice(0, 80), updatedAt: Date.now(), turns: [] };
    state.conversations.unshift(state.current);
  }
  const conv = state.current; // giữ tham chiếu: người dùng có thể chuyển cuộc khác giữa chừng
  const turn = { question, answer: "", standalone: null, sources: [], done: null, error: null, stopped: false };
  conv.turns.push(turn);
  conv.updatedAt = Date.now();
  saveStore();
  renderHistoryList();

  els.welcome.hidden = true;
  if (!retry) addUserMessage(question);
  const view = addBotMessage(turn);
  showInsights(view); // panel chuyển sang câu đang hỏi ngay, hiện "Đang tìm tài liệu…"
  scrollToBottom();

  setBusy(true);
  setStatus("busy", "Đang trả lời…");
  const controller = new AbortController();
  state.controller = controller;
  let rendering = false;
  let timedOut = false;
  let idleTimer;
  const resetIdle = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, IDLE_TIMEOUT_MS);
  };
  resetIdle();

  // Gộp nhiều token vào một lần vẽ mỗi khung hình: mượt hơn và đỡ tốn CPU khi model sinh nhanh.
  const scheduleRender = () => {
    if (rendering) return;
    rendering = true;
    requestAnimationFrame(() => {
      rendering = false;
      // Khung hình này có thể tới sau khi đã kết thúc (done/Dừng): không vẽ lại con trỏ nữa.
      if (!view.bubble || view.finished) return;
      const stick = nearBottom();
      view.bubble.innerHTML = renderAnswer(turn.answer) + '<span class="caret"></span>';
      if (stick) scrollToBottom();
    });
  };

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    for await (const { event, data } of readEvents(response)) {
      resetIdle();
      if (event === "status") {
        view.progress.querySelector(".progress-text").textContent = STEP_TEXT[data.step] ?? "Đang xử lý…";
      } else if (event === "rewrite") {
        turn.standalone = data.standalone;
        addRewrite(view, data.standalone);
      } else if (event === "sources") {
        // Tới trước câu trả lời: người xem thấy hệ thống tìm được gì trong lúc model còn viết.
        turn.sources = view.sources = data.docs;
        if (state.inspected === view) renderSources(view);
      } else if (event === "token") {
        ensureBubble(view);
        turn.answer += data.text;
        scheduleRender();
      } else if (event === "done") {
        turn.done = data;
        finishAnswer(view, turn.answer, data);
        if (state.inspected === view) showInsights(view);
      } else if (event === "error") {
        turn.error = { message: data.message, detail: data.detail };
        showError(view, data.message, data.detail);
      }
    }
  } catch (err) {
    if (err.name === "AbortError" && !timedOut) {
      // Bấm Dừng (hoặc chuyển cuộc khác): giữ phần đã viết, không gửi làm lịch sử vì chưa trọn.
      turn.stopped = true;
      showStopped(view, turn.answer);
    } else {
      const message = timedOut
        ? "Máy chủ phản hồi quá lâu nên mình đã dừng lại. Bạn thử hỏi lại nhé."
        : !navigator.onLine
          ? "Mất kết nối mạng. Kiểm tra mạng rồi thử lại nhé."
          : "Không kết nối được tới server chatbot. Kiểm tra server còn chạy không rồi thử lại nhé.";
      turn.error = { message, detail: String(err) };
      showError(view, message, String(err));
      // Lỗi mạng tới server (không phải timeout, không phải mất mạng): theo dõi tới khi server sống lại.
      if (!timedOut && navigator.onLine) connect();
    }
  } finally {
    clearTimeout(idleTimer);
    // Ghi lượt đã xong (kể cả lỗi/Dừng) để tải lại trang vẫn thấy đúng như lúc rời đi.
    conv.updatedAt = Date.now();
    if (state.conversations.includes(conv)) saveStore(); // cuộc bị xóa giữa chừng thì thôi
    if (state.current === conv) renderHistoryList();
    // Dừng hoặc lỗi: panel đổi "Đang tìm…/Đang đo…" thành trạng thái cuối.
    if (!view.done && state.inspected === view) showInsights(view);
    if (state.controller === controller) state.controller = null;
    setBusy(false);
    if (turn.error && state.current === conv) setStatus("error", "Gặp lỗi");
    if (state.current === conv) scrollToBottom();
  }
}

// ---------------------------------------------------------------------------
// Mở / tạo / xóa cuộc trò chuyện
// ---------------------------------------------------------------------------

function openConversation(conv) {
  state.controller?.abort(); // đang trả lời dở: dừng lại, phần đã viết vẫn được lưu
  closeDrawers(); // điện thoại: chọn xong cuộc trò chuyện thì đóng ngăn kéo lịch sử
  state.current = conv;
  clearInsights();
  // Vẽ lại cuộc đã lưu thì hiện ngay, không cho từng tin nhắn "bay vào" như lúc đang chat.
  document.body.classList.add("restoring");
  requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove("restoring")));
  els.messages.replaceChildren(els.welcome);
  els.welcome.hidden = Boolean(conv?.turns.length);
  let last = null;
  for (const turn of conv?.turns ?? []) last = renderSavedTurn(turn);
  if (last) showInsights(last);
  setStatus("idle", "Sẵn sàng");
  try {
    localStorage.setItem(CURRENT_KEY, conv?.id ?? "");
  } catch {}
  renderHistoryList();
  // Mở cuộc cũ thì nhảy thẳng xuống cuối, không cuộn mượt qua cả cuộc trò chuyện.
  els.messages.style.scrollBehavior = "auto";
  scrollToBottom();
  els.messages.style.scrollBehavior = "";
  els.input.focus();
}

function deleteConversation(conv) {
  if (!confirm(`Xóa cuộc trò chuyện "${conv.title}"?`)) return;
  state.conversations = state.conversations.filter((c) => c !== conv);
  if (state.current === conv) openConversation(null);
  saveStore();
  renderHistoryList();
}

els.newChat.addEventListener("click", () => openConversation(null));
// Nút thùng rác trên header: xóa cuộc đang mở (chưa hỏi gì thì chỉ làm trống).
els.clearChat.addEventListener("click", () => {
  if (state.current) deleteConversation(state.current);
  else openConversation(null);
});

// Mở chatbot ở tab khác cũng lưu vào cùng chỗ: cập nhật danh sách cho khớp.
window.addEventListener("storage", (e) => {
  if (e.key !== STORE_KEY) return;
  const currentId = state.current?.id;
  state.conversations = loadStore();
  // Giữ nguyên cuộc đang mở ở tab này (kể cả đang trả lời dở), chỉ thay phần còn lại.
  if (state.current) {
    state.conversations = state.conversations.filter((c) => c.id !== currentId);
    state.conversations.unshift(state.current);
  }
  renderHistoryList();
});

// Thời gian tương đối ("5 phút") cũ dần: vẽ lại mỗi phút.
setInterval(renderHistoryList, 60_000);

// ---------------------------------------------------------------------------
// Ngăn kéo (tablet: nguồn bên phải; điện thoại: thêm lịch sử bên trái)
// ---------------------------------------------------------------------------

const DRAWER_BUTTONS = { sidebar: els.openSidebar, insights: els.openInsights };
const DRAWER_PANELS = { sidebar: $("#sidebar"), insights: $("#insights") };
let drawerOpener = null; // nút đã mở ngăn kéo: đóng xong trả focus về đó (cho người dùng bàn phím)

function openDrawer(name) {
  closeDrawers({ restoreFocus: false });
  drawerOpener = document.activeElement;
  els.app.classList.add(`show-${name}`);
  els.backdrop.hidden = false;
  DRAWER_BUTTONS[name].setAttribute("aria-expanded", "true");
  if (name === "insights") els.insightsDot.hidden = true;
  // Chờ visibility đổi xong mới focus được vào nút đóng bên trong.
  requestAnimationFrame(() => DRAWER_PANELS[name].querySelector("[data-close-drawer]")?.focus());
}

function closeDrawers({ restoreFocus = true } = {}) {
  const wasOpen = /show-(sidebar|insights)/.test(els.app.className);
  els.app.classList.remove("show-sidebar", "show-insights");
  els.backdrop.hidden = true;
  for (const btn of Object.values(DRAWER_BUTTONS)) btn.setAttribute("aria-expanded", "false");
  if (wasOpen && restoreFocus) drawerOpener?.focus?.();
  drawerOpener = null;
}

els.openSidebar.addEventListener("click", () => openDrawer("sidebar"));
els.openInsights.addEventListener("click", () => openDrawer("insights"));
els.backdrop.addEventListener("click", () => closeDrawers());
document.querySelectorAll("[data-close-drawer]").forEach((b) => b.addEventListener("click", () => closeDrawers()));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDrawers();
});
// Xoay máy / kéo to cửa sổ qua mốc: bỏ trạng thái ngăn kéo để bố cục không kẹt.
for (const mq of Object.values(DRAWER_MQ)) mq.addEventListener("change", () => closeDrawers({ restoreFocus: false }));

// ---------------------------------------------------------------------------
// Kết nối: banner khi server chưa sẵn sàng hoặc mất mạng
// ---------------------------------------------------------------------------

function showBanner(text, kind = "error", { spinner = false } = {}) {
  els.banner.replaceChildren();
  if (spinner) els.banner.append(el("span", "spinner"));
  els.banner.append(el("span", "", text));
  els.banner.dataset.kind = kind;
  els.banner.hidden = false;
}

function hideBanner() {
  els.banner.hidden = true;
}

let infoLoaded = false;
let reconnecting = false; // chỉ chạy một vòng thử lại tại một thời điểm

// Lấy câu hỏi mẫu + cấu hình. Gọi lúc mở trang, và khi câu hỏi gặp lỗi mạng (server bị tắt
// hoặc khởi động lại giữa phiên): khi đó hiện banner và thử lại mỗi vài giây tới khi được.
async function connect() {
  if (reconnecting) return;
  reconnecting = true;
  try {
    const response = await fetch("/api/info");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const info = await response.json();
    hideBanner();
    setStatus("idle", "Sẵn sàng");
    els.metaLine.textContent = `${info.doc_count} tài liệu · ${info.llm ?? info.pipeline}`;
    if (!infoLoaded) {
      for (const q of info.examples) {
        els.welcomeExamples.append(exampleButton(q, "example"));
        els.chips.append(exampleButton(q, "chip"));
      }
      infoLoaded = true;
    }
    reconnecting = false;
  } catch {
    els.metaLine.textContent = "Chưa kết nối được server";
    setStatus("error", "Mất kết nối");
    showBanner("Mất kết nối tới server chatbot. Đang thử kết nối lại… (server khởi động lại cần nạp model)", "info", {
      spinner: true,
    });
    setTimeout(() => {
      reconnecting = false;
      connect();
    }, RECONNECT_MS);
  }
}

window.addEventListener("offline", () => {
  showBanner("Mất kết nối mạng. Câu hỏi sẽ không gửi được cho tới khi có mạng lại.");
  setStatus("error", "Mất mạng");
});
window.addEventListener("online", () => {
  hideBanner();
  if (!state.busy) setStatus("idle", "Sẵn sàng");
});

// ---------------------------------------------------------------------------
// Ô nhập
// ---------------------------------------------------------------------------

function updateCharCount() {
  const n = els.input.value.length;
  els.charCount.hidden = n < CHAR_WARN;
  els.charCount.textContent = `${n}/${MAX_CHARS} ký tự`;
  els.charCount.classList.toggle("over", n >= MAX_CHARS);
}

function autosize() {
  const box = els.input;
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, INPUT_MAX_HEIGHT)}px`;
  box.style.overflowY = box.scrollHeight > INPUT_MAX_HEIGHT ? "auto" : "hidden";
}

function submitInput() {
  const text = els.input.value;
  if (!text.trim()) return;
  els.input.value = "";
  autosize();
  updateSendButton();
  updateCharCount();
  ask(text);
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.busy) state.controller?.abort();
  else submitInput();
});

els.input.addEventListener("keydown", (e) => {
  // isComposing: bộ gõ tiếng Việt (Telex/VNI) đang ghép dấu thì Enter là để chốt chữ, không phải để gửi.
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!state.busy) submitInput();
  }
});

els.input.addEventListener("input", () => {
  autosize();
  updateSendButton();
  updateCharCount();
});

// ---------------------------------------------------------------------------
// Khởi động: mở lại cuộc trò chuyện lần trước, lấy câu hỏi mẫu và thông tin cấu hình
// ---------------------------------------------------------------------------

function exampleButton(text, className) {
  const btn = el("button", className, text);
  btn.type = "button";
  btn.addEventListener("click", () => ask(text));
  return btn;
}

async function init() {
  state.conversations = loadStore();
  let currentId = "";
  try {
    currentId = localStorage.getItem(CURRENT_KEY) ?? "";
  } catch {}
  openConversation(state.conversations.find((c) => c.id === currentId) ?? null);
  await connect();
}

init();
