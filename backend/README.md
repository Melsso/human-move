# chess-backend

FastAPI inference service: loads a checkpoint per rating tier (scanned
from `training/checkpoints/*/best.pt`) and serves moves against them.
Also serves the static frontend at `/`. See the root `README.md`.