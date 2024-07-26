# Use Python 3.9 as the base image
FROM python:3.9

RUN pip install --upgrade pip

# Set up a virtual environment for the Gemini API
ENV GEMINI_ENV=/opt/gemini_env
RUN python -m venv $GEMINI_ENV
ENV PATH="$GEMINI_ENV/bin:$PATH"

# Set up a virtual environment for the YOLO 
ENV YOLO_ENV=/opt/yolo_env
RUN python -m venv $YOLO_ENV
ENV PATH="$YOLO_ENV/bin:$PATH"

# Set up a virtual environment for the TensorFlow Keras project
ENV TENSORFLOW_ENV=/opt/tensorflow_env
RUN python -m venv $TENSORFLOW_ENV
ENV PATH="$TENSORFLOW_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the code into the container
COPY . .

# Install HDF5
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Gemini API dependencies
RUN . $GEMINI_ENV/bin/activate && \
    pip install --no-cache-dir -r gemini_interface/requirements.txt

# Install YOLO dependencies
RUN . $YOLO_ENV/bin/activate && \
    pip install --no-cache-dir -r yolo_interface/requirements.txt

# Install TensorFlow Keras dependencies
RUN . $TENSORFLOW_ENV/bin/activate && \
    pip install --no-cache-dir -r tensorflow_keras_project/requirements.txt

# Exposed ports
EXPOSE 7860

# Default startup command
CMD ["bash"]
