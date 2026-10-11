"use strict";

const $id = (id) => document.getElementById(id);

const START_FEN = new Chess().fen();
const game = new Chess();

let board = null;
let humanColor = "w";
let currentTier = null;
let thinking = false;
let evalEnabled = false;
let resigned = false;
let serverResult = null;

let gameId = 0;
let evalSeq = 0;

let positions = [game.fen()];
let sanHistory = [];
let uciHistory = [];
let viewIndex = 0;

let selectedSquare = null;
let pendingPromotion = null;

const statusEl = $id("status");
const tierSelect = $id("tierSelect");
const colorSelect = $id("colorSelect");
const tierMeta = $id("tierMeta");
const candidatesEl = $id("candidates");
const moveLogEl = $id("moveLog");
const reviewBanner = $id("reviewBanner");

const navStartBtn = $id("navStart");
const navPrevBtn = $id("navPrev");
const navNextBtn = $id("navNext");
const navEndBtn = $id("navEnd");
const takebackBtn = $id("takebackBtn");
const resignBtn = $id("resignBtn");
const retryBtn = $id("retryBtn");

const evalToggle = $id("evalToggle");
const evalHint = $id("evalHint");
const evalBarContainer = $id("evalBarContainer");
const evalBarFill = $id("evalBarFill");
const evalBarLabel = $id("evalBarLabel");

const gameOverModal = $id("gameOverModal");
const modalIcon = $id("modalIcon");
const modalTitle = $id("modalTitle");
const modalSubtitle = $id("modalSubtitle");

const promotionModal = $id("promotionModal");
const promoChoices = $id("promoChoices");

/* ---------- small helpers ---------- */

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "info";
}

function formatDetail(detail) {
  if (typeof detail === "string") return detail;
  try {
    return JSON.stringify(detail);
  } catch (e) {
    return "unknown error";
  }
}

function showRetry(visible) {
  retryBtn.style.display = visible ? "block" : "none";
}

function isOver() {
  return resigned || serverResult !== null || game.game_over();
}

function lastMoverColor() {
  return (uciHistory.length - 1) % 2 === 0 ? "w" : "b";
}

function pliesToTakeBack() {
  return lastMoverColor() === humanColor ? 1 : 2;
}

function canTakeBack() {
  if (uciHistory.length === 0) return false;
  const floor = humanColor === "b" ? 1 : 0;
  return uciHistory.length - pliesToTakeBack() >= floor;
}

function canHumanMove() {
  return (
    !thinking &&
    !isOver() &&
    !pendingPromotion &&
    viewIndex === positions.length - 1 &&
    game.turn() === humanColor
  );
}

function legalMovesFrom(square) {
  return game.moves({ square: square, verbose: true });
}

/* ---------- rendering ---------- */

function showPosition(fen, animate) {
  board.position(fen, animate !== false);
  paintLastMove();
}

function paintLastMove() {
  $("#board .last-move").removeClass("last-move");
  if (viewIndex <= 0) return;
  const uci = uciHistory[viewIndex - 1];
  if (!uci) return;
  $(`#board .square-${uci.slice(0, 2)}, #board .square-${uci.slice(2, 4)}`)
    .addClass("last-move");
}

function highlightLegalMoves(square) {
  legalMovesFrom(square).forEach((move) => {
    const cls =
      move.flags.indexOf("c") !== -1 || move.flags.indexOf("e") !== -1
        ? "capture-square"
        : "highlight-square";
    $(`#board .square-${move.to}`).addClass(cls);
  });
  $(`#board .square-${square}`).addClass("selected-square");
}

function clearHighlights() {
  $("#board .highlight-square, #board .capture-square, #board .selected-square")
    .removeClass("highlight-square capture-square selected-square");
}

function renderCandidates(candidates, chosenUci) {
  candidatesEl.innerHTML = "";

  candidates.forEach((c) => {
    const row = document.createElement("div");
    row.className = "candidate-row" + (c.uci === chosenUci ? " chosen" : "");

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = `${Math.min(100, c.prob * 100).toFixed(1)}%`;

    const label = document.createElement("span");
    label.textContent = c.san;

    const pct = document.createElement("span");
    pct.textContent = `${(c.prob * 100).toFixed(1)}%`;

    row.appendChild(bar);
    row.appendChild(label);
    row.appendChild(pct);
    candidatesEl.appendChild(row);
  });
}

function renderMoveList() {
  moveLogEl.innerHTML = "";

  for (let i = 0; i < sanHistory.length; i += 2) {
    const row = document.createElement("div");
    row.className = "move-row";

    const num = document.createElement("span");
    num.className = "move-num";
    num.textContent = `${i / 2 + 1}.`;
    row.appendChild(num);

    const white = document.createElement("span");
    white.className = "move-entry" + (viewIndex === i + 1 ? " active" : "");
    white.textContent = sanHistory[i];
    white.addEventListener("click", () => goToPosition(i + 1));
    row.appendChild(white);

    if (sanHistory[i + 1]) {
      const black = document.createElement("span");
      black.className = "move-entry" + (viewIndex === i + 2 ? " active" : "");
      black.textContent = sanHistory[i + 1];
      black.addEventListener("click", () => goToPosition(i + 2));
      row.appendChild(black);
    }

    moveLogEl.appendChild(row);
  }

  const active = moveLogEl.querySelector(".active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function updateNavButtons() {
  const atStart = viewIndex === 0;
  const atLive = viewIndex === positions.length - 1;

  navStartBtn.disabled = atStart;
  navPrevBtn.disabled = atStart;
  navNextBtn.disabled = atLive;
  navEndBtn.disabled = atLive;

  if (atLive) {
    reviewBanner.style.display = "none";
  } else {
    reviewBanner.style.display = "block";
    reviewBanner.textContent =
      `Reviewing move ${viewIndex} of ${positions.length - 1} ` +
      `— press End to return`;
  }
}

function updateControls() {
  takebackBtn.disabled = !canTakeBack();
  resignBtn.disabled = isOver() || uciHistory.length === 0;
}

function refresh() {
  renderMoveList();
  updateNavButtons();
  updateControls();
}

/* ---------- eval bar ---------- */

function resetEvalBar() {
  evalBarFill.style.height = "50%";
  evalBarLabel.textContent = "0.0";
}

function updateEvalBar(scoreCp, mate) {
  let whiteFraction;
  let label;

  if (mate !== null && mate !== undefined) {
    whiteFraction = mate > 0 ? 1 : 0;
    label = `M${Math.abs(mate)}`;
  } else if (scoreCp !== null && scoreCp !== undefined) {
    whiteFraction = 0.5 + 0.5 * Math.tanh(scoreCp / 400);
    const pawns = scoreCp / 100;
    label = (pawns > 0 ? "+" : "") + pawns.toFixed(1);
  } else {
    whiteFraction = 0.5;
    label = "";
  }

  evalBarFill.style.height = `${(whiteFraction * 100).toFixed(1)}%`;
  evalBarLabel.textContent = label;
}

async function fetchEval(fen) {
  if (!evalEnabled) return;
  const seq = ++evalSeq;

  try {
    const res = await fetch("/api/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen }),
    });
    if (!res.ok) return;

    const data = await res.json();
    if (seq !== evalSeq || !evalEnabled) return;

    if (!data.available) {
      evalEnabled = false;
      evalToggle.checked = false;
      evalToggle.disabled = true;
      evalBarContainer.classList.remove("visible");
      evalHint.textContent =
        "stockfish not found on this machine -- see backend/README.md to enable the eval bar";
      return;
    }

    updateEvalBar(data.score_cp, data.mate);
  } catch (e) {
    /* eval is optional, ignore failures */
  }
}

evalToggle.addEventListener("change", () => {
  evalEnabled = evalToggle.checked;

  if (evalEnabled) {
    evalBarContainer.classList.add("visible");
    resetEvalBar();
    fetchEval(positions[viewIndex]);
  } else {
    evalSeq++;
    evalBarContainer.classList.remove("visible");
  }
});

/* ---------- tiers ---------- */

function updateTierMeta(t) {
  tierMeta.textContent =
    `epoch ${t.epoch} · val top1 ${(t.val_top1 * 100).toFixed(1)}% · ` +
    `val top3 ${(t.val_top3 * 100).toFixed(1)}%`;
}

async function loadTiers() {
  try {
    const res = await fetch("/api/tiers");
    const tiers = await res.json();

    if (tiers.length === 0) {
      setStatus(
        "no checkpoints found -- train a model or run `make fetch-weights`",
        "error",
      );
      tierSelect.disabled = true;
      return;
    }

    tierSelect.innerHTML = "";
    tiers.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.tier;
      opt.textContent = `Tier ${t.tier}`;
      opt.dataset.epoch = t.epoch;
      opt.dataset.top1 = t.val_top1;
      opt.dataset.top3 = t.val_top3;
      tierSelect.appendChild(opt);
    });

    currentTier = tiers[0].tier;
    updateTierMeta(tiers[0]);
    setStatus("your move", "info");
  } catch (e) {
    setStatus(`could not reach the server: ${e}`, "error");
  }
}

tierSelect.addEventListener("change", () => {
  currentTier = tierSelect.value;
  const opt = tierSelect.selectedOptions[0];
  updateTierMeta({
    epoch: opt.dataset.epoch,
    val_top1: Number(opt.dataset.top1),
    val_top3: Number(opt.dataset.top3),
  });
});

/* ---------- navigation ---------- */

function goToPosition(index) {
  const clamped = Math.max(0, Math.min(index, positions.length - 1));
  if (clamped === viewIndex) return;

  viewIndex = clamped;
  selectedSquare = null;
  clearHighlights();

  showPosition(positions[viewIndex]);
  refresh();
  fetchEval(positions[viewIndex]);
}

navStartBtn.addEventListener("click", () => goToPosition(0));
navPrevBtn.addEventListener("click", () => goToPosition(viewIndex - 1));
navNextBtn.addEventListener("click", () => goToPosition(viewIndex + 1));
navEndBtn.addEventListener("click", () => goToPosition(positions.length - 1));

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (pendingPromotion) cancelPromotion();
    else hideGameOverModal();
    return;
  }

  if (pendingPromotion || gameOverModal.classList.contains("visible")) return;
  if (document.activeElement && document.activeElement.tagName === "SELECT") {
    return;
  }

  if (e.key === "ArrowLeft") {
    e.preventDefault();
    goToPosition(viewIndex - 1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    goToPosition(viewIndex + 1);
  } else if (e.key === "Home") {
    e.preventDefault();
    goToPosition(0);
  } else if (e.key === "End") {
    e.preventDefault();
    goToPosition(positions.length - 1);
  }
});

/* ---------- game over ---------- */

function outcomeInfo() {
  if (resigned) {
    return { kind: "loss", title: "You resigned", subtitle: "", icon: "🏳️" };
  }

  if (game.in_checkmate()) {
    const humanWon = game.turn() !== humanColor;
    const mated = game.turn() === "w" ? "white" : "black";
    return {
      kind: humanWon ? "win" : "loss",
      title: humanWon ? "You won!" : "You lost",
      subtitle: `Checkmate — ${mated} is mated`,
      icon: humanWon ? "🏆" : "😔",
    };
  }

  const draw = (subtitle) => ({
    kind: "draw",
    title: "Draw",
    subtitle,
    icon: "🤝",
  });

  if (game.in_stalemate()) return draw("Stalemate");
  if (game.in_threefold_repetition()) return draw("Threefold repetition");
  if (game.insufficient_material()) return draw("Insufficient material");
  if (game.in_draw()) return draw("50-move rule");

  if (serverResult === "1/2-1/2") {
    return draw("Repetition or 50-move rule");
  }
  if (serverResult === "1-0" || serverResult === "0-1") {
    const humanWon = (serverResult === "1-0") === (humanColor === "w");
    return {
      kind: humanWon ? "win" : "loss",
      title: humanWon ? "You won!" : "You lost",
      subtitle: "",
      icon: humanWon ? "🏆" : "😔",
    };
  }

  return null;
}

function finishGame() {
  const info = outcomeInfo();
  if (!info) return;

  setStatus(
    `game over: ${info.title}${info.subtitle ? " — " + info.subtitle : ""}`,
    "info",
  );

  modalIcon.textContent = info.icon;
  modalTitle.textContent = info.title;
  modalTitle.className = `modal-title ${info.kind}`;
  modalSubtitle.textContent = info.subtitle;
  gameOverModal.classList.add("visible");
  updateControls();
}

function hideGameOverModal() {
  gameOverModal.classList.remove("visible");
}

$id("modalNewGameBtn").addEventListener("click", () => {
  hideGameOverModal();
  newGame();
});
$id("modalCloseBtn").addEventListener("click", hideGameOverModal);
gameOverModal.addEventListener("click", (e) => {
  if (e.target === gameOverModal) hideGameOverModal();
});

/* ---------- promotion picker ---------- */

function openPromotion(from, to) {
  pendingPromotion = { from, to };
  promoChoices.innerHTML = "";

  ["Q", "R", "B", "N"].forEach((piece) => {
    const btn = document.createElement("button");
    btn.className = "secondary";
    btn.title = { Q: "Queen", R: "Rook", B: "Bishop", N: "Knight" }[piece];

    const img = document.createElement("img");
    img.src = `vendor/pieces/${humanColor}${piece}.png`;
    img.alt = btn.title;
    btn.appendChild(img);

    btn.addEventListener("click", () => choosePromotion(piece.toLowerCase()));
    promoChoices.appendChild(btn);
  });

  promotionModal.classList.add("visible");
}

function closePromotionModal() {
  promotionModal.classList.remove("visible");
}

function choosePromotion(piece) {
  if (!pendingPromotion) return;
  const { from, to } = pendingPromotion;
  pendingPromotion = null;
  closePromotionModal();
  playHuman(from, to, piece);
}

function cancelPromotion() {
  pendingPromotion = null;
  closePromotionModal();
  showPosition(game.fen(), false);
}

$id("promoCancelBtn").addEventListener("click", cancelPromotion);
promotionModal.addEventListener("click", (e) => {
  if (e.target === promotionModal) cancelPromotion();
});

/* ---------- move flow ---------- */

function popPlies(count) {
  for (let i = 0; i < count; i++) {
    game.undo();
    uciHistory.pop();
    positions.pop();
    sanHistory.pop();
  }
  viewIndex = positions.length - 1;
  showPosition(game.fen(), false);
  refresh();
}

function playHuman(from, to, promotion) {
  const fenBefore = game.fen();
  const historyBefore = uciHistory.slice();

  const move = game.move({ from, to, promotion });
  if (move === null) {
    showPosition(game.fen(), false);
    return;
  }

  const uci = from + to + (move.promotion || "");
  uciHistory.push(uci);
  positions.push(game.fen());
  sanHistory.push(move.san);
  viewIndex = positions.length - 1;

  showPosition(game.fen());
  refresh();
  fetchEval(game.fen());

  if (isOver()) {
    finishGame();
    return;
  }

  exchange(uci, fenBefore, historyBefore);
}

function applyModelMove(data) {
  const uci = data.model_move_uci;
  const applied = game.move({
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  });

  if (applied === null) {
    throw new Error("the model's move is not legal on the local board");
  }

  const wasAtLive = viewIndex === positions.length - 1;

  uciHistory.push(uci);
  positions.push(game.fen());
  sanHistory.push(data.model_move_san);

  if (wasAtLive) {
    viewIndex = positions.length - 1;
    showPosition(game.fen());
  }

  refresh();
  renderCandidates(data.top_candidates, uci);
  fetchEval(positions[viewIndex]);
}

async function exchange(humanUci, fenBefore, historyBefore) {
  if (!currentTier) {
    setStatus("no opponent available", "error");
    return;
  }

  const myGame = gameId;
  let applied = false;

  thinking = true;
  showRetry(false);
  setStatus("model is thinking...", "info");

  try {
    const body = { fen: fenBefore, history: historyBefore, tier: currentTier };
    if (humanUci) body.move_uci = humanUci;

    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (myGame !== gameId) return;

    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail;
      } catch (e) {
        /* keep statusText */
      }
      if (myGame !== gameId) return;

      if (humanUci) popPlies(1);
      else showRetry(true);
      setStatus(`error: ${formatDetail(detail)}`, "error");
      return;
    }

    const data = await res.json();
    if (myGame !== gameId) return;

    if (data.model_move_uci) {
      await sleep(250);
      if (myGame !== gameId) return;
      applyModelMove(data);
      applied = true;
    }

    if (data.game_over || game.game_over()) {
      serverResult = data.game_over ? data.result : null;
      finishGame();
    } else {
      setStatus("your move", "info");
    }
  } catch (e) {
    if (myGame !== gameId) return;

    if (!applied) {
      if (humanUci) popPlies(1);
      else showRetry(true);
    }
    setStatus(`error: ${e}`, "error");
  } finally {
    if (myGame === gameId) {
      thinking = false;
      updateControls();
    }
  }
}

retryBtn.addEventListener("click", () => exchange(null, START_FEN, []));

/* ---------- buttons ---------- */

function takeBack() {
  if (!canTakeBack()) return;

  gameId++;
  thinking = false;
  resigned = false;
  serverResult = null;
  pendingPromotion = null;
  selectedSquare = null;
  closePromotionModal();
  hideGameOverModal();
  showRetry(false);
  clearHighlights();
  candidatesEl.innerHTML = "";

  popPlies(pliesToTakeBack());
  setStatus("your move", "info");
  fetchEval(positions[viewIndex]);
}

function resign() {
  if (isOver() || uciHistory.length === 0) return;
  if (!window.confirm("Resign this game?")) return;

  gameId++;
  thinking = false;
  resigned = true;
  showRetry(false);
  refresh();
  finishGame();
}

function newGame() {
  gameId++;
  thinking = false;
  resigned = false;
  serverResult = null;
  pendingPromotion = null;
  selectedSquare = null;
  closePromotionModal();
  hideGameOverModal();
  showRetry(false);

  let choice = colorSelect.value;
  if (choice === "random") choice = Math.random() < 0.5 ? "white" : "black";
  humanColor = choice === "black" ? "b" : "w";

  game.reset();
  positions = [game.fen()];
  sanHistory = [];
  uciHistory = [];
  viewIndex = 0;

  board.orientation(humanColor === "w" ? "white" : "black");
  evalBarContainer.classList.toggle("flipped", humanColor === "b");
  clearHighlights();
  showPosition(game.fen(), false);

  candidatesEl.innerHTML = "";
  refresh();

  if (evalEnabled) {
    resetEvalBar();
    fetchEval(game.fen());
  }

  if (humanColor === "b") exchange(null, START_FEN, []);
  else setStatus("your move", "info");
}

$id("newGameBtn").addEventListener("click", newGame);
takebackBtn.addEventListener("click", takeBack);
resignBtn.addEventListener("click", resign);

colorSelect.addEventListener("change", () => {
  const humanMoves = uciHistory.length - (humanColor === "b" ? 1 : 0);
  if (humanMoves <= 0 || isOver()) newGame();
});

/* ---------- board interaction ---------- */

function onDragStart(source, piece) {
  if (!canHumanMove()) return false;
  if (piece.charAt(0) !== humanColor) return false;

  selectedSquare = source;
  clearHighlights();
  highlightLegalMoves(source);
  return true;
}

function onDrop(source, target) {
  if (source === target) return "snapback";

  clearHighlights();
  selectedSquare = null;

  if (!canHumanMove()) return "snapback";

  const candidates = legalMovesFrom(source).filter((m) => m.to === target);
  if (candidates.length === 0) return "snapback";

  if (candidates[0].promotion) {
    openPromotion(source, target);
    return;
  }

  playHuman(source, target);
}

function onSnapEnd() {
  if (pendingPromotion) return;
  board.position(game.fen());
}

function onSquareClick(square) {
  if (!canHumanMove()) return;

  const piece = game.get(square);

  if (selectedSquare) {
    if (square === selectedSquare) {
      selectedSquare = null;
      clearHighlights();
      return;
    }

    const targets = legalMovesFrom(selectedSquare).filter(
      (m) => m.to === square,
    );

    if (targets.length > 0) {
      const from = selectedSquare;
      selectedSquare = null;
      clearHighlights();

      if (targets[0].promotion) openPromotion(from, square);
      else playHuman(from, square);
      return;
    }

    clearHighlights();
    if (piece && piece.color === humanColor) {
      selectedSquare = square;
      highlightLegalMoves(square);
    } else {
      selectedSquare = null;
    }
    return;
  }

  if (piece && piece.color === humanColor) {
    selectedSquare = square;
    highlightLegalMoves(square);
  }
}

board = Chessboard("board", {
  draggable: true,
  position: "start",
  pieceTheme: "vendor/pieces/{piece}.png",
  onDragStart,
  onDrop,
  onSnapEnd,
});

$("#board").on("click", ".square-55d63", function () {
  const square = $(this).data("square");
  if (square) onSquareClick(square);
});

new ResizeObserver(() => {
  board.resize();
  selectedSquare = null;
  clearHighlights();
  paintLastMove();
}).observe($id("board"));

refresh();
loadTiers();