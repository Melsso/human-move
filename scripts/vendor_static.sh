#!/usr/bin/env bash
# Downloads the UI's third-party assets into backend/static/vendor so the app
# runs fully offline. Run once, commit the result.
set -euo pipefail

DEST="backend/static/vendor"
mkdir -p "$DEST/pieces"

CDN="https://cdnjs.cloudflare.com/ajax/libs"
curl -fsSL "$CDN/jquery/3.7.1/jquery.min.js" -o "$DEST/jquery.min.js"
curl -fsSL "$CDN/chess.js/0.10.2/chess.js" -o "$DEST/chess.js"
curl -fsSL "$CDN/chessboard-js/1.0.0/chessboard-1.0.0.min.js" -o "$DEST/chessboard.min.js"
curl -fsSL "$CDN/chessboard-js/1.0.0/chessboard-1.0.0.min.css" -o "$DEST/chessboard.min.css"

for piece in wK wQ wR wB wN wP bK bQ bR bB bN bP; do
  curl -fsSL "https://chessboardjs.com/img/chesspieces/wikipedia/${piece}.png" \
    -o "$DEST/pieces/${piece}.png"
done

echo "vendored into $DEST"