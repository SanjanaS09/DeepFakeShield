# DeepFakeShield Deployment Guide

## Best Platform Choice

For this repository as it exists today:

- Frontend: `Vercel` or `Netlify`
- Backend: `Render` using Docker for CPU inference
- GPU upgrade path: `Runpod`, `AWS EC2 GPU`, or `Google Cloud GPU VMs`

This project is not a good fit for serverless backend hosting because the backend is a long-running Flask app with heavy ML dependencies, large uploads, and model loading at startup.

## What I Found In This Repo

- The frontend is a Create React App project in `deepfakeshield/`.
- The backend is a Flask app in `backend/`.
- The backend expects trained checkpoints under `backend/checkpoints/...`, but those model files are not in the repo.
- The repo contains a local training dataset under `backend/dataset/` that is about `10.8 GB`.
- The backend dependency set is much more compatible with `Python 3.10` than modern default cloud Python versions like `3.14`.

## Recommended Deployment Path

### Option A: Cleanest Setup

- Deploy the frontend on `Vercel` or `Netlify`
- Deploy the backend on `Render` as a Docker web service
- Store your trained model checkpoints outside Git and mount or copy them into the backend at deploy time

This is the best balance of simplicity, cost, and reliability for your current architecture.

### Option B: One Platform

- Deploy both frontend and backend on `Render`

This is easier to manage in one dashboard, but the frontend experience is usually nicer on Vercel or Netlify.

### Option C: GPU / Faster Inference

- Deploy the frontend on `Vercel` or `Netlify`
- Deploy the backend on `Runpod`, `AWS EC2 GPU`, or `Google Cloud GPU VMs`

Use this if image inference is slow on CPU or if you want video/audio inference to feel production-ready.

## Before You Deploy

1. Do not deploy the `backend/dataset/` folder.
2. Do not deploy local virtualenvs like `backend/.venv/` or `backend/deepfake_env/`.
3. Make sure your trained files exist for:
   - `CHECKPOINTS_DIR/image/best_model.pth`
   - `CHECKPOINTS_DIR/video/best_model.pth`
   - `CHECKPOINTS_DIR/audio/best_model.pth`
4. Use `Python 3.10.x` for the backend.
   The repo now includes `backend/.python-version` with `3.10.11`.
5. Set the frontend environment variable `REACT_APP_API_BASE_URL` to your deployed backend URL.

## Frontend Deployment

Directory: `deepfakeshield/`

### Vercel

- Root directory: `deepfakeshield`
- Build command: `npm run build`
- Output directory: `build`
- Environment variable:
  - `REACT_APP_API_BASE_URL=https://your-backend-url`

The repo now includes `deepfakeshield/vercel.json` so React Router routes rewrite to `index.html`.

### Netlify

- Base directory: `deepfakeshield`
- Build command: `npm run build`
- Publish directory: `build`
- Environment variable:
  - `REACT_APP_API_BASE_URL=https://your-backend-url`

The repo now includes `deepfakeshield/netlify.toml` for SPA routing.

## Backend Deployment

Directory: `backend/`

### Render

Create a new Web Service and choose Docker.

- Root directory: `backend`
- Dockerfile: `backend/Dockerfile`
- Install source: `backend/requirements-deploy.txt` is already wired into the Docker build
- Health check path: `/`

Set these environment variables:

- `HOST=0.0.0.0`
- `PORT=10000`
- `FLASK_ENV=production`
- `DEVICE=cpu`
- `SECRET_KEY=...`
- `JWT_SECRET_KEY=...`
- `CHECKPOINTS_DIR=/app/checkpoints`

Optional:

- `DATABASE_URL=sqlite:///deepfake_detection.db`
- `REDIS_URL=...`

Important:

- Render services use an ephemeral filesystem by default. If you want checkpoints to stay on-disk across deploys, use a persistent disk or bake them into the image.
- The included Docker image starts Gunicorn with one worker intentionally, so the ML models are not loaded multiple times in memory.

### Railway

Railway can also deploy the backend Dockerfile.

- Service source: repo
- Root directory: `backend`
- Dockerfile path: `backend/Dockerfile` or use `backend` as the service root
- Set the same environment variables as above

## Model Files Strategy

Because the trained weights are not in this repo, pick one of these approaches:

1. Put the checkpoints into `backend/checkpoints/` before building the backend image.
2. Mount persistent storage and set `CHECKPOINTS_DIR` to that mount path.
3. Download the checkpoints from object storage during startup.

For this codebase, option `1` or `2` is simplest.

## Chrome Extension

The browser extension in `extension/` is deployed separately from the website:

- local load for testing: Chrome Extensions page with Developer Mode
- production distribution: Chrome Web Store package submission

It is not part of the Vercel/Netlify/Render web deployment flow.

## Practical Recommendation

If you want the easiest real deployment right now, do this:

1. Push a cleaned repo to GitHub without the dataset and local virtualenvs.
2. Deploy `deepfakeshield/` to Vercel.
3. Deploy `backend/` to Render with Docker.
4. Upload or mount the checkpoint files.
5. Set `REACT_APP_API_BASE_URL` in Vercel to the Render backend URL.

If you want, the next step can be setting up a ready-to-push `render.yaml` or `Railway` config for you.
