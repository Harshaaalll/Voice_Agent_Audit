# Deploying (free tier)

> **Status: written, NOT executed.** No cloud account, `gcloud` run or Hugging Face push was done while writing this.
> The local Docker build was verified earlier; these cloud steps were not. Expect to adjust flags.
> Check each provider's current free-tier limits before relying on them.

The image listens on port **8000** and defaults to the mock backend with SQLite (data is lost when the instance restarts).
That is fine for a demo; for kept results use an external Postgres with pgvector (for example a free serverless tier)
and set `DATABASE_URL`.

## Option A: GCP Cloud Run (the "cloud" line for the resume)
1. **Set a budget alert first.** Console -> Billing -> Budgets & alerts -> create a budget of a few rupees with email alerts.
   Cloud Run scales to zero, so an idle service should cost nothing, but a mistake elsewhere will not.
2. Project setup (once):
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
3. Deploy from the repo root (Cloud Build uses the `Dockerfile`):
   ```bash
   gcloud run deploy voice-agent-audit --source . --region asia-south1 --port 8000 \
     --allow-unauthenticated --max-instances 1 --memory 1Gi \
     --set-env-vars VAA_BACKEND=mock,VAA_MODEL=mock
   ```
   Keep `--max-instances 1` so a traffic spike cannot run up cost.
4. For real model calls, store the key as a secret, never as a plain env var:
   ```bash
   printf '%s' "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
   gcloud run services update voice-agent-audit --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
     --update-env-vars VAA_BACKEND=gemini,VAA_MODEL=gemini-2.0-flash
   ```
5. Check: `curl https://<service-url>/health`.
6. The best retrieval setup (`VAA_EMBEDDER=fastembed VAA_RETRIEVAL=hybrid`) downloads a ~220 MB model on first start,
   so give the instance 2 GiB of memory and expect a slow cold start. The default (`hash`, `dense`) needs no download.
7. Redis, the worker and async jobs are not part of this deploy. `/evaluate` runs synchronously without `REDIS_URL`.
8. Teardown: `gcloud run services delete voice-agent-audit --region asia-south1`.

## Option B: Hugging Face Spaces (no card)
1. Create a new Space, SDK **Docker**, blank template.
2. In the Space repository, put this repo's files and replace its `README.md` with `deploy/hf-space-README.md`
   (the front matter below tells Spaces which port to use).
3. Add `GEMINI_API_KEY` under Settings -> Secrets if you want real model calls. Spaces sleep when idle and have small CPU/RAM.

## Cost and safety checklist
- Budget alert set, `--max-instances 1`, no always-on instance.
- No secret in the image, the repo, or `--set-env-vars`.
- The service has no authentication. Do not put real borrower data in it. Data here is synthetic.
