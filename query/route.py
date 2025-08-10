import os
from flask import Flask, request, jsonify
from rag import query
from ingestPdf import embedPdf
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

@app.route('/hackrx/run', methods=['POST'])
def question():
    if request.is_json:
        data = request.get_json()
        document_url = data.get("documents")
        question_list = data.get("questions")

        embedPdf(document_url)

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        answer_list = []
        with open("log.txt", "a") as file:
            file.write(f"Request at {timestamp}\n")
            file.write("Responses\n\n")
            file.write(f"Document link : {document_url}\n\n")
            for ques in question_list:
                ans = query(ques)
                answer_list.append(ans)
                file.write(f"Ques : {ques}\n")
                file.write(f"Ans : {ans}\n\n")
            file.write("----------------------------------------------------------------------------------------\n")

        return jsonify({"answers": answer_list})

    else:
        return jsonify({"error" : "Request must be JSON"})

if __name__ == '__main__':
    app.run(debug=True)