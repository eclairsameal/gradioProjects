# Gradio Project

[gradio](https://www.gradio.app/)

## Install Project Docker Environment

```bash
docker build -t gradio_env:latest .

docker run -it --name gradio_env_projects -p 7880:7880 -d -v $(pwd):/app gradio_env /bin/bash
```

## Gemini Project

[Gemini API](https://ai.google.dev/?gad_source=1&gclid=CjwKCAjww_iwBhApEiwAuG6ccFMl8dbbBnP8jX9QP7KdmJTG1z269c11KuOsG2vYaIOWtLayUTp_zBoCuPgQAvD_BwE)

Create a file named config.ini in the gemini_env folder.

Please enter your own Gemini API KEY in config.ini.

config.ini：
```
[DEFAULT]
GOOGLE_API_KEY = Gemini API KEY
```

### Start the virtual environment
```bash
source /opt/gemini_env/bin/activate
```
### Execute the program
```bash
cd gemini_interface

python main.py
```


## YOLO Project

[YOLO](https://docs.ultralytics.com/zh)

Confidence threshold:

IoU（Intersection over Union）threshold:

### Start the virtual environment 
```bash
source /opt/yolo_env/bin/activate
```
### Execute the program
```bash

cd yolo_interface

python main.py
```

### Error
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
Solutions：
```
pip install opencv-python-headless
```

## Tensorflow Project

### Start the virtual environment and execute the program
```bash
source /opt/tensorflow_env/bin/activate

python tensorflow_keras_project/main.py
```

### Error
RuntimeError: Failed to import transformers.models.distilbert.modeling_tf_distilbert because of the following error (look up to see its traceback):
Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. Please install the backwards-compatible tf-keras package with `pip install tf-keras`.

Solutions：
```
pip install tf-keras
```
