PYTHON = ./venv/bin/python
PIP    = ./venv/bin/pip

deploy-all: kind-cluster ambassador seldon-core prometheus grafana

kind-cluster:
	kind create cluster --name seldon-cluster --config kind-cluster.yaml --image=kindest/node:v1.21.2

ambassador: kind-cluster
	helm repo add datawire https://www.getambassador.io
	helm repo update
	helm upgrade --install ambassador datawire/ambassador \
		--set image.repository=docker.io/datawire/ambassador \
		--set service.type=ClusterIP \
		--set replicaCount=1 \
		--set crds.keep=false \
		--set enableAES=false \
		--create-namespace \
		--namespace ambassador \
		--version 6.9.5

seldon-core: kind-cluster
	helm upgrade --install seldon-core seldon-core-operator \
		--repo https://storage.googleapis.com/seldon-charts \
		--set crd.create=true \
		--set usageMetrics.enabled=true \
		--set ambassador.enabled=true \
		--create-namespace \
		--namespace seldon-system \
		--version 1.15

prometheus:
	helm upgrade --install seldon-monitoring kube-prometheus \
		--version 8.9.1 \
		--set fullnameOverride=seldon-monitoring \
		--create-namespace \
		--namespace seldon-monitoring \
		--repo https://charts.bitnami.com/bitnami

grafana:
	helm repo add grafana https://grafana.github.io/helm-charts
	helm upgrade --install grafana-seldon-monitoring grafana/grafana \
		--version 6.56.1 \
		--values grafana.yaml \
		--namespace seldon-monitoring

venv:
	python -m venv venv

train: venv
	$(PIP) install -r requirements.txt
	$(PYTHON) train.py

build-solo: train
	$(PYTHON) edit_model_settings.py solo
	$(PYTHON) -m mlserver build . -t alph8rd/uplift-predictor

build-two: train
	$(PYTHON) edit_model_settings.py two
	$(PYTHON) -m mlserver build . -t alph8rd/uplift-predictor

push-image:
	docker push alph8rd/uplift-predictor

deploy-uplift-predictor:
	kubectl apply -f uplift-deploy.yaml

delete-uplift-predictor:
	kubectl delete -f uplift-deploy.yaml

podmonitor:
	kubectl apply -f podmonitor.yaml

predictor-forward-port:
	kubectl port-forward svc/uplift-predictor-uplift-predictor-uplift-predictor 9000:9000

prom-forward-port:
	kubectl port-forward svc/seldon-monitoring-prometheus 9090:9090 --namespace seldon-monitoring

graf-forward-port:
	kubectl get secret --namespace seldon-monitoring grafana-seldon-monitoring -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
	kubectl port-forward svc/grafana-seldon-monitoring 3000:80 --namespace seldon-monitoring

test: venv
	$(PIP) install requests pytest
	$(PYTHON) -m pytest tests/
