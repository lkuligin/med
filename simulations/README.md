
export REGION=europe-west4
export PROJECT_ID=
gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/patient-sim/iris-sim-chat:latest"


gcloud beta run deploy <SERVICE_NAME> \
  --image=<REGION>-docker.pkg.dev/<PROJECT_ID>/<AR_REPO>/<SERVICE_NAME>:latest \
  --region=<REGION> \
  --min-instances=0 \
  --max-instances=1 \
  --cpu=1 \
  --memory=512Mi \
  --cpu-throttling \
  --no-allow-unauthenticated \
  --iap \
  --service-account=patient-sim-runner@<PROJECT_ID>.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=<PROJECT_ID>,GCS_BUCKET=<GCS_BUCKET>,GCS_CHAT_PREFIX=chats,MODEL_NAME=gemini-3.8-flash,IAP_AUDIENCE=<AUDIENCE>"

