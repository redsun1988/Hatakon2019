__author__ = 'xead'
from codecs import open
import time
from flask import Flask, render_template, request
from TextGenerator import TextGenerator
from StringProcessor import StringProcessor

app = Flask(__name__)

print("Preparing model")
start_time = time.time()

textGenerator = TextGenerator(
        './data/encoder-model-test.h5',
        './data/decoder-model-test.h5',
        './data/hyper-test.json')

print("Model is ready")
print(time.time() - start_time, "seconds")

@app.route("/", methods=["POST", "GET"])
def index_page(questionText="a dummy question"):
   subject = "New Cool subject"
   if request.method == "POST":
     questionText = request.form["questionText"]
     print(questionText)
     
     #predict a subject
     questionText = StringProcessor.Clean(questionText)
     subject = textGenerator.predict(questionText)
     subject = StringProcessor.RemoveLoops(subject)
     print(subject)

   return render_template('hello.html', questionText=questionText, subject=subject)
if __name__ == "__main__":
   app.run()
