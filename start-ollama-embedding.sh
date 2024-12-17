#!/bin/sh

EMBED_MODEL_DIR="/root/.ollama/models/manifests/registry.ollama.ai/library/mxbai-embed-large"

export OLLAMA_HOST=0.0.0.0:11435
ollama serve &

echo 'Waiting for Ollama service to start...'
sleep 30

if [ ! "$(ls -A $EMBED_MODEL_DIR)" ]; then
    echo 'mxbai-embed-large model not found, downloading...'
    ollama pull mxbai-embed-large
    echo 'mxbai-embed-large downloaded successfully.'
else
    echo 'mxbai-embed-large model already present, skipping download.'
fi

# Keep the container running
tail -f /dev/null
