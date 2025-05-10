mkdir logs
touch app.log

mkdir models
Download a LLM model, change name to llm-model.bin

python -m venv venv
# Windows:
myenv\Scripts\activate

# Linux/Mac:
source myenv/bin/activate

pip install -r requirements.txt
