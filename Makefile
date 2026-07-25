SHELL := /bin/bash
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHONPATH := $(ROOT)
SCENE ?= room
INSTRUCTION ?= Go to the red bin, avoid the chair, inspect the table, then return home.
SORT_INSTRUCTION ?= Sort every red item into the red bucket and every blue item into the blue bucket.
ALLOW_PROXY ?= 0
ALLOW_KINEMATIC ?= 0
ALLOW_RULE_PLANNER ?= 0

export PYTHONPATH

.PHONY: setup-brev runpod-setup runpod-lobby runpod-play runpod-validate mac-setup mac-smoke mac-demo demo-3 demo-3-scripted demo-3-validate mac-osm-build mac-osm mac-golden-gate-build mac-golden-gate mac-salesforce-park-build mac-salesforce-park replit-worker replit-test check-gpu check-isaac check-livestream download-g1 download-g1-mjcf gemini-sort-demo build live-demo recorded-demo api vla dashboard eval test integration-test clean

runpod-setup:
	bash scripts/setup_runpod.sh

runpod-lobby:
	bash scripts/run_g1_demo_5_runpod.sh lobby

runpod-play:
	bash scripts/run_g1_demo_5_runpod.sh play

runpod-validate:
	bash scripts/run_g1_demo_5_runpod.sh validate

mac-setup:
	bash scripts/setup_mac.sh

download-g1-mjcf:
	bash scripts/download_g1_mjcf.sh

mac-smoke:
	.venv-mac/bin/python -m pathvla.mujoco_sorting_demo --headless --validate-only

mac-demo:
	bash scripts/run_mac_demo.sh "$(SORT_INSTRUCTION)"

demo-3:
	bash scripts/run_g1_demo_3.sh match

demo-3-scripted:
	bash scripts/run_g1_demo_3.sh scripted

demo-3-validate:
	bash scripts/run_g1_demo_3.sh validate

mac-osm-build:
	.venv-mac/bin/python -m pathvla.osm_mujoco --build-only

mac-osm:
	bash scripts/run_mujoco_osm.sh

mac-golden-gate-build: mac-osm-build

mac-golden-gate: mac-osm

mac-salesforce-park-build:
	.venv-mac/bin/python -m pathvla.osm_mujoco --config config/osm_sf_salesforce_park.yaml --output-dir outputs/mujoco_sf_salesforce_park --build-only

mac-salesforce-park:
	OSM_SCENE_CONFIG=config/osm_sf_salesforce_park.yaml OSM_OUTPUT_DIR=outputs/mujoco_sf_salesforce_park bash scripts/run_mujoco_osm.sh

replit-worker:
	bash scripts/run_replit_worker.sh

replit-test:
	.venv-mac/bin/python -m pytest tests/test_replit_worker.py -q

download-g1:
	bash scripts/download_g1_usd.sh

gemini-sort-demo:
	bash scripts/run_gemini_sorting_demo.sh "$(SORT_INSTRUCTION)"

setup-brev:
	bash scripts/setup_brev.sh

check-gpu:
	bash scripts/check_gpu.sh

check-isaac:
	bash scripts/check_isaac.sh

check-livestream:
	bash scripts/check_livestream.sh

build:
	docker compose -f docker/docker-compose.yaml build

live-demo:
	bash scripts/run_live_demo.sh "$(SCENE)" "$(INSTRUCTION)" "$(ALLOW_PROXY)" "$(ALLOW_KINEMATIC)" "$(ALLOW_RULE_PLANNER)"

recorded-demo:
	bash scripts/run_headless_recorded_demo.sh "$(SCENE)" "$(INSTRUCTION)" "$(ALLOW_PROXY)" "$(ALLOW_KINEMATIC)" "$(ALLOW_RULE_PLANNER)"

api:
	bash scripts/start_api.sh

vla:
	bash scripts/start_vla.sh

dashboard:
	bash scripts/start_dashboard.sh

eval:
	bash scripts/run_eval.sh

test:
	python3 -m pytest tests -m "not integration"

integration-test:
	python3 -m pytest tests -m integration

clean:
	find outputs -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
