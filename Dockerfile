FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt install --no-install-recommends -y python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /rag-project

# Copy the current directory contents into the container at /rag-project
COPY requirements.txt /rag-project/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt