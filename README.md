# CropDoctor AI Backend (Render)

FastAPI service for maize and sugarcane photo classification.

- `POST /predict` (multipart form: `image`, `crop` = `maize` or `sugarcane`)
- `GET /health`

Models: `Julians30/maize-disease-models` (Apache-2.0) and
`dwililiya/sugarcane-plant-diseases-classification` (CDLA-Sharing-1.0).
No API key is needed.

## Deploy on Render
1. Push this folder to a GitHub repository.
2. Render -> New -> Web Service -> pick the repo. Language: Docker. Instance type: Free.
3. Add environment variable `ALLOWED_ORIGINS` = your Vercel URL (optional at first).
4. After it is live, open `https://YOUR-SERVICE.onrender.com/health`.
5. Use `https://YOUR-SERVICE.onrender.com/predict` as BACKEND_URL in index.html.

AI decision support only. Confirm with an agricultural extension officer before treating crops.
