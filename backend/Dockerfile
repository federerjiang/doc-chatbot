# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./app /code/app
COPY requirements.txt /code/
COPY ./assets /code/assets
COPY ./assets-test /code/assets-test


# CMD sudo apt update
# CMD sudo apt install build-essential
# Copy the script into the image
# COPY setenv.sh /code/setenv.sh

# Make the script executable and run it
# RUN chmod +x /code/setenv.sh && /bin/bash /code/setenv.sh
# Install any needed packages specified in requirements.txt
# RUN apt-get update && \
    # apt-get install -y gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    git \
    curl \
    cmake \               
    libmecab-dev \        
    mecab-ipadic-utf8 \ 
    # Clean up the apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/root/.cargo/bin
RUN curl https://sh.rustup.rs -sSf > /rust.sh && sh /rust.sh -y \
    && rustup install stable
# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade pip
RUN pip install --no-cache-dir setuptools_rust
RUN pip install Cython
# RUN pip install --no-cache-dir --no-build-isolation sudachipy
# RUN pip install Cython && pip install SudachiPy
# 尝试不同的方式安装 SudachiPy
RUN pip install --no-cache-dir sudachipy || \
    pip install --no-cache-dir --no-build-isolation sudachipy || \
    pip install --no-cache-dir sudachipy==0.6.6

    # 安装 spaCy
RUN pip install --no-cache-dir spacy
# 下载和安装最大的日语模型
# RUN python -m spacy download ja_core_news_lg
RUN pip install -U ginza ja_ginza

# RUN apt-get install libgl1-mesa-glx -y
# RUN apt-get install libglib2.0-0 -y 
# RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
    # pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run app.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
