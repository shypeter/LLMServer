from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from llm_handler import LLMHandler

app = Flask(__name__)
CORS(app)

model_path = os.getenv('MODEL_PATH', './models/llm-model.bin')
log_path = os.getenv('LOG_PATH', './logs/app.log')
# 設置日誌記錄
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

llm_handler = LLMHandler(model_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/compet', methods=['POST'])
def competition():
    data = request.json
    query = data.get('query', '')
    logger.info(query)
    if not query:
        return jsonify({"error": "Query is required"}), 400

    #logger.info(context)
    answer = llm_handler.generate_answer(query)
    return jsonify({
        "answer": answer,
    })

if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)