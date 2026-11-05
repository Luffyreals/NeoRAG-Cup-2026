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

// Các bước server báo qua sự kiện `status`, hiện thành danh sách có đồng hồ trong bong bóng chờ.
const STEP_TEXT = {
  followup: "Hiểu ngữ cảnh câu hỏi",
  retrieve: "Tìm tài liệu liên quan",
  generate: "Viết câu trả lời",
};
const STEP_TICK_MS = 100; // đồng hồ của bước đang chạy cập nhật 10 lần/giây
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
  examples: [], // câu hỏi mẫu (từ /api/info)
  suggestions: {}, // bảng gợi ý hỏi tiếp: { CLB: { chủ đề: [câu hỏi] } }
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

// ---------------------------------------------------------------------------
// Markdown cho câu trả lời
// Nguyên tắc an toàn: thoát HTML cho TOÀN BỘ văn bản trước, sau đó mới chèn thẻ do mình tạo.
// Văn bản từ model không bao giờ thành thẻ HTML, kể cả khi model viết <script> hay link javascript:.
// ---------------------------------------------------------------------------

// Link markdown [chữ](url), URL đầy đủ, hoặc tên miền trần như ttv.proptit.com / fb.com/proptit.
// Chỉ nhận http(s), nên "javascript:..." không bao giờ thành link.
const LINK_RE = new RegExp(
  [
    String.raw`\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)`, // 1: chữ, 2: url
    String.raw`(https?:\/\/[^\s<]+)`, // 3
    String.raw`\b([a-z0-9._%+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,})\b`, // 4: email (đứng trước tên miền trần)
    String.raw`\b((?:www\.)?(?:[a-z0-9-]+\.)+(?:com|vn|net|org|io|me|dev|app|edu)(?:\.vn)?\b(?:\/[^\s<]*)?)`, // 5
  ].join("|"),
  "gi",
);
// Dấu câu cuối câu dính vào URL ("…tại ttv.proptit.com.") không thuộc về link.
// Văn bản đã thoát HTML nên dấu nháy lúc này là &quot; / &#39;.
const TRAILING_PUNCT_RE = /(?:[.,;:!?)\]…]|&quot;|&#39;)+$/;

// Số liệu người dùng hay cần: ngày tháng (09/10/2011, 20/08, 8/2026), giờ (19h30), số tiền, phần trăm.
const FACT_RE = new RegExp(
  [
    String.raw`\b\d{1,2}\/\d{1,2}(?:\/\d{2,4})?\b`,
    String.raw`\b\d{1,2}\/\d{4}\b`,
    String.raw`\b\d{1,2}h(?:\d{2})?\b`,
    String.raw`\b\d{1,3}(?:[.,]\d{3})+\s?(?:đ|đồng|VNĐ|VND)`,
    String.raw`\b\d+\s?(?:k|nghìn|triệu)\s?(?:đ|đồng)?(?=[\s.,;)]|$)`,
    String.raw`\b\d+(?:[.,]\d+)?\s?%`,
  ].join("|"),
  "g",
);

function linkTag(url, label) {
  const href = /^https?:\/\//i.test(url) ? url : `https://${url}`;
  // url đã được thoát HTML từ trước nên đặt thẳng vào thuộc tính href là an toàn.
  return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
}

// Định dạng trong một dòng (đã thoát HTML): link, **đậm**, số liệu.
// Cắt chuỗi theo link trước, để **đậm**/số liệu không chạm vào bên trong href.
function renderInline(escaped) {
  let html = "";
  let last = 0;
  for (const m of escaped.matchAll(LINK_RE)) {
    const [, mdText, mdUrl, fullUrl, email, bareUrl] = m;
    html += formatText(escaped.slice(last, m.index));
    if (mdText) {
      html += linkTag(mdUrl, formatText(mdText));
    } else if (email) {
      html += `<a href="mailto:${email}">${email}</a>`;
    } else {
      const raw = fullUrl ?? bareUrl;
      const tail = raw.match(TRAILING_PUNCT_RE)?.[0] ?? "";
      const url = raw.slice(0, raw.length - tail.length);
      html += linkTag(url, url) + tail;
    }
    last = m.index + m[0].length;
  }
  return html + formatText(escaped.slice(last));
}

function formatText(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(FACT_RE, '<span class="fact">$&</span>');
}

const UL_RE = /^\s*[-*•]\s+(.*)$/;
const OL_RE = /^\s*(\d{1,2})[.)]\s+(.*)$/;
const HEADING_RE = /^\s*#{1,6}\s+(.*)$/;
const CARET = '<span class="caret"></span>';

// Chia câu trả lời thành khối: đoạn văn (cách nhau bằng dòng trống), danh sách gạch đầu dòng,
// danh sách đánh số, tiêu đề. Các dòng liền nhau trong một đoạn nối bằng <br>.
// caret = true khi đang stream: con trỏ nằm cuối chữ cuối cùng, không rơi xuống dòng riêng.
function renderAnswer(text, { caret = false } = {}) {
  const blocks = [];
  let para = [];
  let list = null; // { tag: "ul" | "ol", start, items: [] }
  const flushPara = () => {
    if (para.length) blocks.push(`<p>${para.map(renderInline).join("<br>")}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) {
      const start = list.tag === "ol" && list.start !== 1 ? ` start="${list.start}"` : "";
      blocks.push(`<${list.tag}${start}>${list.items.map((i) => `<li>${renderInline(i)}</li>`).join("")}</${list.tag}>`);
    }
    list = null;
  };

  for (const line of escapeHtml(text.trim()).split("\n")) {
    const ul = line.match(UL_RE);
    const ol = !ul && line.match(OL_RE);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (list?.tag !== tag) {
        flushList();
        list = { tag, start: ol ? Number(ol[1]) : 1, items: [] };
      }
      list.items.push(ul ? ul[1] : ol[2]);
    } else if (!line.trim()) {
      flushPara();
      flushList();
    } else if (HEADING_RE.test(line)) {
      flushPara();
      flushList();
      blocks.push(`<p class="md-heading">${renderInline(line.match(HEADING_RE)[1])}</p>`);
    } else if (list && /^\s{2,}\S/.test(line)) {
      list.items[list.items.length - 1] += ` ${line.trim()}`; // dòng thụt lề = phần tiếp của mục trước
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();

  const html = blocks.join("");
  if (!caret) return html;
  if (!html) return CARET;
  // Chèn con trỏ ngay trước thẻ đóng của khối cuối (</p> hoặc </li></ul>).
  return html.replace(/<\/(?:p|li)>(?:<\/[ou]l>)?$/, (end) => CARET + end);
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

// ---------------------------------------------------------------------------
// Tô sáng bằng chứng: chỗ nào trong tài liệu trùng với câu trả lời.
// So khớp ngay trên trình duyệt, không gọi thêm LLM, không đổi câu trả lời.
//   - "phrase": cụm >= PHRASE_N tiếng liên tiếp giống hệt câu trả lời (câu được chép từ tài liệu)
//   - "key": từ có chữ số hoặc link/email xuất hiện trong câu trả lời (ngày, số tiền, link...)
// ---------------------------------------------------------------------------

// Một "từ" = chuỗi chữ/số, giữ liền các dấu bên trong: 09/10/2011, ttv.proptit.com, a@b.com.
const WORD_RE = /[\p{L}\p{N}]+(?:[./:@_-][\p{L}\p{N}]+)*/gu;
// 5 tiếng liên tiếp: đủ dài để không khớp nhầm cụm phổ biến như "CLB Lập Trình PTIT" (4 tiếng).
const PHRASE_N = 5;
const SNIPPET_LEAD = 60; // khi đoạn trích nhảy tới chỗ khớp, giữ ~60 ký tự phía trước làm ngữ cảnh

function tokenize(text) {
  return [...text.matchAll(WORD_RE)].map((m) => ({
    w: m[0].toLowerCase(),
    // Tiếng viết hoa đầu thường là tên riêng (CLB, Lập, Trình, PTIT): lặp lại khắp tài liệu.
    upper: m[0][0] !== m[0][0].toLowerCase(),
    start: m.index,
    end: m.index + m[0].length,
  }));
}

const isKeyToken = (w) => (/\d/.test(w) && w.length >= 2) || (/[.@]/.test(w) && /[a-z]/.test(w) && w.length >= 6);

// Chuẩn bị từ câu trả lời một lần: tập các cụm PHRASE_N tiếng và tập từ khóa.
function evidenceIndex(answer) {
  const words = tokenize(answer.normalize("NFC")).map((t) => t.w);
  const grams = new Set();
  for (let i = 0; i + PHRASE_N <= words.length; i++) grams.add(words.slice(i, i + PHRASE_N).join(" "));
  return { grams, keys: new Set(words.filter(isKeyToken)) };
}

// Trả về các khoảng [start, end) trong text cần tô, đã gộp các từ liền nhau cùng loại.
function findEvidence(text, index) {
  const toks = tokenize(text);
  const kind = new Array(toks.length).fill(null);
  for (let i = 0; i + PHRASE_N <= toks.length; i++) {
    const run = toks.slice(i, i + PHRASE_N);
    // Cụm gần như toàn tên riêng ("CLB Lập Trình PTIT được") trùng vì tên CLB, không phải vì chép ý:
    // cần ít nhất 3 tiếng thường.
    if (run.filter((t) => !t.upper).length < 3) continue;
    if (index.grams.has(run.map((t) => t.w).join(" "))) kind.fill("phrase", i, i + PHRASE_N);
  }
  toks.forEach((t, i) => {
    if (!kind[i] && index.keys.has(t.w)) kind[i] = "key";
  });
  const ranges = [];
  toks.forEach((t, i) => {
    if (!kind[i]) return;
    const prev = ranges.at(-1);
    // Gộp với khoảng trước nếu từ ngay trước cũng được tô cùng loại (kể cả dấu cách/dấu phẩy ở giữa).
    if (prev && prev.last === i - 1 && prev.kind === kind[i]) {
      prev.end = t.end;
      prev.last = i;
    } else {
      ranges.push({ start: t.start, end: t.end, kind: kind[i], last: i });
    }
  });
  return ranges;
}

// Dựng đoạn text[from, to) thành DOM, bọc các khoảng khớp bằng <mark>.
// Dùng text node chứ không dùng innerHTML: nội dung tài liệu không bao giờ thành thẻ HTML.
function paintEvidence(text, ranges, from = 0, to = text.length) {
  const frag = document.createDocumentFragment();
  if (from > 0) frag.append("…");
  let pos = from;
  for (const r of ranges) {
    if (r.end <= from || r.start >= to) continue;
    const s = Math.max(r.start, from);
    const e = Math.min(r.end, to);
    if (s > pos) frag.append(text.slice(pos, s));
    const mark = el("mark", `evidence ${r.kind}`, text.slice(s, e));
    frag.append(mark);
    pos = e;
  }
  if (pos < to) frag.append(text.slice(pos, to));
  if (to < text.length) frag.append("…");
  return frag;
}

// Đoạn trích mặc định là phần đầu tài liệu. Nếu chỗ khớp đầu tiên nằm ngoài phần đó,
// trích quanh chỗ khớp: người xem thấy ngay câu trả lời lấy từ đâu mà không phải mở toàn văn.
function snippetWindow(text, ranges, length) {
  // Ưu tiên cụm câu chữ trùng; một con số lẻ (như "2026") ở đầu bài chưa chắc là chỗ được dùng.
  const first = ranges.find((r) => r.kind === "phrase") ?? ranges[0];
  if (!first || first.end <= length) return [0, Math.min(length, text.length)];
  let from = Math.max(0, first.start - SNIPPET_LEAD);
  const space = text.lastIndexOf(" ", from);
  if (from > 0 && space !== -1 && first.start - space < SNIPPET_LEAD * 2) from = space + 1; // không cắt giữa từ
  return [from, Math.min(text.length, from + length)];
}

function scrollToFirstMark(box) {
  const mark = box.querySelector("mark");
  if (!mark) return;
  const top = mark.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
  box.scrollTop = Math.max(0, top - 24); // chỉ cuộn trong ô toàn văn, không cuộn cả trang
}

function sourceCard(doc, index, evidence) {
  const card = el("article", "source");
  card.dataset.club = doc.label;
  card.dataset.index = index;
  const head = el("div", "source-head");
  const badge = el("span", "badge", doc.label);
  badge.dataset.club = doc.label;
  head.append(el("span", "source-num", String(index + 1)), badge, el("span", "source-topic", doc.topic));
  head.querySelector(".source-topic").title = doc.topic;

  const fullText = doc.text.normalize("NFC");
  const ranges = evidence ? findEvidence(fullText, evidence) : [];
  if (ranges.length) {
    const count = el("span", "evidence-count", `${ranges.length} chỗ khớp`);
    count.title = "Số chỗ trong tài liệu này trùng với câu trả lời";
    head.append(count);
    card.classList.add("has-evidence");
  }
  // Độ dài đoạn trích lấy theo server (bỏ dấu "…" ở cuối).
  const snippetLen = doc.snippet.replace(/…$/, "").length;
  const [from, to] = snippetWindow(fullText, ranges, snippetLen);

  const text = el("p", "source-snippet");
  text.append(paintEvidence(fullText, ranges, from, to));
  card.append(head, text);
  if (from > 0 || to < fullText.length) {
    // Mở/thu toàn văn ngay trong thẻ: không cần hộp thoại, không mất ngữ cảnh.
    const toggle = el("button", "link-btn", "Xem toàn văn");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      text.replaceChildren(open ? paintEvidence(fullText, ranges, from, to) : paintEvidence(fullText, ranges));
      text.classList.toggle("full", !open);
      toggle.textContent = open ? "Xem toàn văn" : "Thu gọn";
      toggle.setAttribute("aria-expanded", String(!open));
      if (!open) scrollToFirstMark(text); // mở toàn văn: nhảy tới chỗ khớp đầu tiên
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
  // Chỉ so khớp khi câu trả lời đã xong (đang stream thì câu còn dở, kết quả nhảy liên tục).
  const answer = view.done && view.turn?.answer;
  const evidence = answer ? evidenceIndex(answer) : null;
  docs.forEach((doc, i) => body.append(sourceCard(doc, i, evidence)));
  const scored = Boolean(evidence && body.querySelector(".has-evidence"));
  body.classList.toggle("scored", scored); // làm mờ nhẹ các nguồn không có chỗ nào khớp
  if (scored) {
    const note = el("p", "evidence-note");
    note.append(el("mark", "evidence phrase", "Tô vàng"), ": câu chữ trùng với câu trả lời · ");
    note.append(el("mark", "evidence key", "tô tím"), ": số liệu, link");
    body.prepend(note);
  }
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
  const progress = el("ol", "progress");
  // Dòng tạm trước khi server báo bước đầu tiên (thường chỉ vài chục ms).
  const pending = el("li", "step active");
  pending.append(el("span", "spinner"), el("span", "step-label", "Đang gửi câu hỏi…"));
  progress.append(pending);
  body.append(progress);
  msg.append(logoAvatar(), body);
  els.messages.append(msg);
  // view = phần giao diện của một lượt; view.sources/done dùng chung với turn để panel phải đọc.
  return {
    msg, body, progress, bubble: null, turn, question: turn.question, sources: turn.sources, done: null,
    step: null, // bước đang chạy: { li, time, start }
    ticker: 0,
  };
}

// Đánh dấu bước đang chạy là xong: spinner thành ✓, đồng hồ dừng ở thời gian đo được.
function completeStep(view) {
  const step = view.step;
  if (!step) return;
  step.li.classList.replace("active", "done");
  step.li.querySelector(".spinner").replaceWith(el("span", "step-icon", "✓"));
  step.time.textContent = `${((performance.now() - step.start) / 1000).toFixed(1)}s`;
  view.step = null;
}

// Server báo sang bước mới: chốt bước trước, thêm dòng mới với đồng hồ chạy từ 0.
function startStep(view, key) {
  completeStep(view);
  view.progress.querySelector(".step:not(.done):not([data-step])")?.remove(); // bỏ dòng "Đang gửi…"
  const li = el("li", "step active");
  li.dataset.step = key;
  const time = el("span", "step-time", "0.0s");
  time.setAttribute("aria-hidden", "true"); // không để trình đọc màn hình đọc số nhảy liên tục
  li.append(el("span", "spinner"), el("span", "step-label", STEP_TEXT[key] ?? "Đang xử lý"), time);
  view.progress.append(li);
  view.step = { li, time, start: performance.now() };
  // Một bộ đếm cho cả lượt, chỉ cập nhật bước đang chạy.
  if (!view.ticker) {
    view.ticker = setInterval(() => {
      if (view.step) view.step.time.textContent = `${((performance.now() - view.step.start) / 1000).toFixed(1)}s`;
    }, STEP_TICK_MS);
  }
}

// Kết thúc lượt (xong, Dừng, lỗi): tắt đồng hồ và bỏ danh sách bước.
// Thời gian chính xác từng bước (đo ở server) vẫn còn ở panel Hiệu năng.
function stopProgress(view) {
  clearInterval(view.ticker);
  view.ticker = 0;
  view.step = null;
  view.progress.remove();
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
  stopProgress(view);
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
  stopProgress(view);
  if (answer) ensureBubble(view).innerHTML = renderAnswer(answer);
  view.body.append(el("div", "msg-meta", "Đã dừng"));
}

function finishAnswer(view, answer, done) {
  view.finished = true;
  view.done = done;
  ensureBubble(view);
  stopProgress(view);
  view.bubble.innerHTML = renderAnswer(answer);
  if (done.no_info) {
    view.msg.classList.add("no-info");
    view.bubble.prepend(el("span", "tag", "Ngoài phạm vi tài liệu"));
  } else if (view.sources.length) {
    view.body.append(sourceChips(view));
  }
  view.body.append(actionBar(view, answer, done));
}

// ---------------------------------------------------------------------------
// Thanh thao tác dưới câu trả lời: thời gian · Sao chép · Sửa câu hỏi · 👍 · 👎
// ---------------------------------------------------------------------------

const ICONS = {
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16v4Z"/><path d="M13.5 6.5l4 4"/>',
  up: '<path d="M7 11v9H4v-9h3Zm0 0 4-8a2.5 2.5 0 0 1 2.5 2.5V9h5.2a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.5 20H7"/>',
  down: '<path d="M7 13V4H4v9h3Zm0 0 4 8a2.5 2.5 0 0 0 2.5-2.5V15h5.2a2 2 0 0 0 2-2.3l-1.2-7A2 2 0 0 0 17.5 4H7"/>',
};

function metaButton(icon, label, { text, className = "" } = {}) {
  const btn = el("button", `meta-btn ${className}`.trim());
  btn.type = "button";
  btn.innerHTML = `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[icon]}</svg>`;
  if (text) btn.append(el("span", "", text));
  // Luôn có tên cho trình đọc màn hình, kể cả khi nhãn chữ bị ẩn ở màn hẹp.
  btn.setAttribute("aria-label", label);
  btn.title = label;
  return btn;
}

// Bỏ ký hiệu markdown khi chép: dán vào Zalo/Messenger không còn dấu ** thừa.
const plainAnswer = (answer) => answer.replace(/\*\*(.+?)\*\*/g, "$1").replace(/^\s*[-*]\s+/gm, "• ").trim();

// Clipboard API chỉ chạy ở trang "an toàn" (https hoặc localhost) và khi trang đang được focus.
// Lúc demo có thể mở qua IP mạng LAN (http://192.168...), nên dự phòng bằng cách chép kiểu cũ.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = el("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.cssText = "position:fixed;opacity:0;pointer-events:none";
    document.body.append(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    area.remove();
    return ok;
  }
}

function actionBar(view, answer, done) {
  const meta = el("div", "msg-meta");
  const time = el("span", "meta-time", `${done.total.toFixed(1)}s`);
  time.title = "Tổng thời gian trả lời";
  meta.append(time);

  const copy = metaButton("copy", "Sao chép câu trả lời", { text: "Sao chép" });
  copy.addEventListener("click", async () => {
    const label = copy.querySelector("span");
    const ok = await copyText(plainAnswer(answer));
    copy.focus(); // cách chép dự phòng làm mất focus: trả lại cho nút
    if (ok) {
      copy.querySelector("svg").innerHTML = ICONS.check;
      label.textContent = "Đã chép";
    } else {
      label.textContent = "Không chép được";
    }
    setTimeout(() => {
      copy.querySelector("svg").innerHTML = ICONS.copy;
      label.textContent = "Sao chép";
    }, 1500);
  });

  // Đưa câu hỏi về ô nhập để sửa rồi gửi lại: hữu ích khi câu trả lời chưa đúng ý vì hỏi chưa rõ.
  const edit = metaButton("edit", "Sửa câu hỏi rồi gửi lại", { text: "Sửa câu hỏi" });
  edit.addEventListener("click", () => {
    if (state.busy) return;
    els.input.value = view.question;
    els.input.dispatchEvent(new Event("input")); // cập nhật chiều cao, bộ đếm ký tự, nút Gửi
    els.input.focus();
    els.input.setSelectionRange(view.question.length, view.question.length);
  });

  meta.append(copy, edit, feedbackButtons(view));
  return meta;
}

// 👍/👎 lưu vào lượt hỏi (localStorage) cùng lịch sử. Bấm lại lần nữa để bỏ chọn.
function feedbackButtons(view) {
  const group = el("span", "feedback");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Đánh giá câu trả lời");
  const thanks = el("span", "feedback-thanks", "Cảm ơn góp ý!");
  thanks.setAttribute("role", "status");
  thanks.hidden = true;
  const buttons = {
    up: metaButton("up", "Câu trả lời hữu ích", { className: "fb-btn" }),
    down: metaButton("down", "Câu trả lời chưa tốt", { className: "fb-btn" }),
  };
  const paint = () => {
    for (const [value, btn] of Object.entries(buttons)) {
      const on = view.turn.feedback === value;
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-pressed", String(on));
    }
  };
  for (const [value, btn] of Object.entries(buttons)) {
    btn.dataset.value = value;
    btn.addEventListener("click", () => {
      view.turn.feedback = view.turn.feedback === value ? null : value;
      saveStore();
      paint();
      thanks.hidden = !view.turn.feedback;
      clearTimeout(view.thanksTimer);
      view.thanksTimer = setTimeout(() => (thanks.hidden = true), 2000);
    });
    group.append(btn);
  }
  paint(); // mở lại cuộc trò chuyện cũ: hiện đúng lựa chọn đã lưu
  group.append(thanks);
  return group;
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
      view.bubble.innerHTML = renderAnswer(turn.answer, { caret: true });
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
        startStep(view, data.step);
        if (nearBottom()) scrollToBottom(); // danh sách dài thêm một dòng
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
    if (state.current === conv) {
      refreshChips(); // gợi ý "hỏi tiếp" theo câu vừa trả lời (lỗi/Dừng: về câu mẫu)
      scrollToBottom();
    }
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
  refreshChips();
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
// ---------------------------------------------------------------------------
// Phím tắt
//   Esc        đóng ngăn kéo; không có ngăn kéo nào mở và bot đang trả lời thì Dừng
//   Ctrl/⌘+K   cuộc trò chuyện mới
//   /          nhảy vào ô nhập (khi chưa đang gõ ở đâu)
// ---------------------------------------------------------------------------

const isTyping = (node) => node?.matches?.("input, textarea, select, [contenteditable]");
const drawerOpen = () => els.app.classList.contains("show-sidebar") || els.app.classList.contains("show-insights");

document.addEventListener("keydown", (e) => {
  if (e.isComposing) return; // bộ gõ tiếng Việt đang ghép dấu
  if (e.key === "Escape") {
    if (drawerOpen()) closeDrawers();
    else if (state.busy) state.controller?.abort(); // giống bấm nút Dừng
    return;
  }
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
    e.preventDefault(); // Ctrl+K của trình duyệt là ô tìm kiếm: ở đây dùng cho cuộc mới
    openConversation(null);
    return;
  }
  if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey && !isTyping(document.activeElement)) {
    e.preventDefault(); // không gõ dấu "/" vào ô nhập
    closeDrawers({ restoreFocus: false });
    els.input.focus();
  }
});

// ---------------------------------------------------------------------------
// Theme sáng/tối (lưu trong trình duyệt; index.html đọc lại trước khi vẽ để không bị chớp)
// ---------------------------------------------------------------------------

const THEME_KEY = "neorag.theme";
const themeToggle = $("#theme-toggle");

function applyTheme(theme) {
  const light = theme === "light";
  if (light) document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  if (!themeToggle) return; // index.html cũ còn trong cache: thiếu nút thì bỏ qua, không làm hỏng cả trang
  themeToggle.setAttribute("aria-pressed", String(light));
  const label = light ? "Chuyển sang nền tối" : "Chuyển sang nền sáng (máy chiếu)";
  themeToggle.setAttribute("aria-label", label);
  themeToggle.title = label;
}

themeToggle?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {} // chế độ ẩn danh / bị chặn lưu: vẫn đổi được, chỉ không nhớ lần sau
});
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
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
      for (const q of info.examples) els.welcomeExamples.append(exampleButton(q, "example"));
      state.examples = info.examples;
      state.suggestions = info.suggestions ?? {};
      infoLoaded = true;
      refreshChips(); // cuộc trò chuyện cũ đã mở trước khi có bảng gợi ý: vẽ lại hàng chip
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

// ---------------------------------------------------------------------------
// Gợi ý "hỏi tiếp" theo ngữ cảnh
// Sau mỗi câu trả lời, lấy CLB + chủ đề của các nguồn vừa dùng rồi tra bảng gợi ý của server
// (app/suggestions.py). Không gọi LLM nên hiện ngay, và câu gợi ý đã được thử là trả lời được.
// ---------------------------------------------------------------------------

const MAX_FOLLOWUPS = 3;
const normQ = (q) => q.toLowerCase().replace(/[?!.\s]+/g, " ").trim();

function followupQuestions(conv) {
  const last = conv?.turns.at(-1);
  if (!last?.done || last.done.no_info || !last.sources?.length) return [];
  const table = state.suggestions;
  // Câu đã hỏi trong cuộc này (cả câu gốc lẫn câu đã viết lại) thì không gợi ý lại.
  const asked = new Set(conv.turns.flatMap((t) => [t.question, t.standalone ?? ""]).filter(Boolean).map(normQ));

  // Thứ tự chủ đề: nguồn có chỗ khớp với câu trả lời (đúng là đang nói về nó) lên trước, rồi theo hạng.
  const evidence = evidenceIndex(last.answer);
  const docs = [...last.sources].sort(
    (a, b) =>
      Number(findEvidence(b.text.normalize("NFC"), evidence).length > 0) -
      Number(findEvidence(a.text.normalize("NFC"), evidence).length > 0),
  ); // sort ổn định: cùng nhóm thì giữ thứ hạng retrieval
  const keys = [];
  for (const d of docs) {
    if (table[d.label]?.[d.topic] && !keys.some((k) => k.club === d.label && k.topic === d.topic)) {
      keys.push({ club: d.label, topic: d.topic });
    }
  }
  if (!keys.length) return [];

  const picked = [];
  const take = (club, q) => {
    if (picked.length < MAX_FOLLOWUPS && !asked.has(normQ(q)) && !picked.some((p) => p.q === q)) {
      picked.push({ club, q });
      return true;
    }
    return false;
  };
  const fresh = (club, topic) => (table[club]?.[topic] ?? []).filter((q) => !asked.has(normQ(q)));

  // 1) Mỗi chủ đề một câu, tối đa 2 câu: đi rộng trước khi đi sâu.
  for (const k of keys) {
    if (picked.length >= MAX_FOLLOWUPS - 1) break;
    const q = fresh(k.club, k.topic)[0];
    if (q) take(k.club, q);
  }
  // 2) Cùng chủ đề nhưng ở CLB khác: "MPC tuyển thế nào?" → gợi ý "GCC tuyển thế nào?" để so sánh.
  const top = keys[0];
  for (const club of Object.keys(table)) {
    if (club === top.club) continue;
    const q = fresh(club, top.topic)[0];
    if (q && take(club, q)) break;
  }
  // 3) Còn chỗ: câu thứ hai của các chủ đề trên.
  for (const k of keys) for (const q of fresh(k.club, k.topic)) take(k.club, q);
  return picked;
}

// Hàng chip phía trên ô nhập: câu mẫu (cuộc mới) hoặc "Hỏi tiếp" (sau một câu trả lời).
function refreshChips() {
  const followups = followupQuestions(state.current);
  els.chips.replaceChildren();
  els.chips.classList.toggle("followups", followups.length > 0);
  if (!followups.length) {
    for (const q of state.examples) els.chips.append(exampleButton(q, "chip"));
    return;
  }
  els.chips.append(el("span", "chips-label", "Hỏi tiếp"));
  for (const { club, q } of followups) {
    const chip = exampleButton(q, "chip followup");
    chip.dataset.club = club; // chấm màu CLB trước câu hỏi
    els.chips.append(chip);
  }
  els.chips.scrollLeft = 0;
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
