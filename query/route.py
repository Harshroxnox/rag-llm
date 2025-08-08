import os
from flask import Flask, request, jsonify
from rag import query_response

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

@app.route('/hackrx/run', methods=['POST'])
def question():
    if request.is_json:
        data = request.get_json()
        # document_link = data.get("documents")
        question_list = data.get("questions")

        answer_list = []
        for ques in question_list:
            ans = query_response(ques)
            answer_list.append(ans)

        return jsonify({"answers": answer_list})

    else:
        return jsonify({"error" : "Request must be JSON"})

if __name__ == '__main__':
    app.run(debug=True)