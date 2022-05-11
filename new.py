from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
model_name = 'smilegate-ai/kor_unsmile'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = TextClassificationPipeline(
    model = model,
    tokenizer = tokenizer,
    device = -1,   # cpu: -1, gpu: gpu number
    return_all_scores = True,
    function_to_apply = 'sigmoid'
)

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/detect', methods=["POST"])
def detect():
    text = request.json["text"]
    return jsonify(pipe(text))

if __name__ == "__main__":
    app.run('127.0.0.1', 5050, False)