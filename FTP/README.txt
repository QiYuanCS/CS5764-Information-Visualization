gcloud auth application-default login
cloudresourcemanager.googleapis.com
pip install -r requirements.txt
gcloud auth configure-docker
gcloud services enable containerregistry.googleapis.com


Python3 -m venv .venv
. ./.venv/bin/activate


docker build -f Dockerfile -t gcr.io/new-project-383706/my_app:latest .
docker push gcr.io/new-project-383706/ftp_app:latest
gcloud run deploy dashapp --image gcr.io/new-project-383706/ftp_app:latest
