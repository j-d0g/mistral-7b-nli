#!/bin/bash

# Create a temp file to install Jupyter and then run the notebook server
cat > run_jupyter_temp.sh << 'EOF'
#!/bin/bash
# Install Jupyter inside the container
pip install jupyter

# Start Jupyter Notebook server
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root --no-browser
EOF

# Make it executable
chmod +x run_jupyter_temp.sh

# Run your Docker container with the modified script
docker run --gpus device=0 --rm -it \
  -p 8888:8888 \
  -v $(pwd):/app \
  -w /app \
  mistral-nli-ft \
  /bin/bash -c "/app/run_jupyter_temp.sh"