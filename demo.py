__author__ = 'xead'
from codecs import open
import time
from flask import Flask, render_template, request
from textGenerator import TextGenerator
app = Flask(__name__)

print("Preparing model")
start_time = time.time()
#textGenerator = TextGenerator('./data/FinalResult.csv', 'Text', 'Subject')
textGenerator = TextGenerator('./data/news_summary_more.csv', 'text', 'headlines')
print("Model is ready")
print(time.time() - start_time, "seconds")

@app.route("/", methods=["POST", "GET"])
def index_page(questionText="a dummy question"):
   subject = "New Cool subject"
   if request.method == "POST":
     questionText = request.form["questionText"]
     print(questionText)
     #predict a subject
     subject = textGenerator.predict(questionText)
     print(subject)
   return render_template('hello.html', questionText=questionText, subject=subject)
if __name__ == "__main__":
   app.run()
