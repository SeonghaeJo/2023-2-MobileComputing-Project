from flask import Flask, request
from decoder import load_and_get_model, inference
app = Flask(__name__)


vocabs = []
with open("vocab.txt", 'r') as f:
  for _ in range(5000):
    vocabs.append(f.readline().replace("\n", ""))
print("Prepare decoder model...")
model = load_and_get_model()
print("Decoder model loaded successfully!")

@app.route("/")
def hello_world():
  return "hello world!"

@app.route("/decode", methods=["POST"])
def run_decoder():
  data = request.json
  print("Run decoder inference...")
  result = inference(model, data, vocabs)
  print(f"Inference result : {result}")
  return result

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8123)
