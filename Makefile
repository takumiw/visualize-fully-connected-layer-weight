FOLDER=$$(pwd)
IMAGE_NAME=visualize-fc-weight:latest

.PHONY: build-gpu
build-gpu: # Build docker image
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} . -f Dockerfile_gpu

.PHONY: start-gpu
start-gpu: build-gpu # Start docker container
	echo "Starting container ${IMAGE_NAME}"
	docker run --gpus all --rm -it -v ${FOLDER}:/work -w /work -p 9001:9001 ${IMAGE_NAME}

.PHONY: jupyter
jupyter:
	poetry run jupyter notebook --port 9001 --ip=0.0.0.0 --allow-root