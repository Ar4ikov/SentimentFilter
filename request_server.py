# | Created by Ar4ikov
# | Время: 16.02.2020 - 19:56

from ssl import SSLContext

import sentiment_filter as s
from flask import Flask, request, jsonify

DEBUG = False
HOST = "localhost"
PORT = 80

USE_SSL = False
CERTS = ("/path/to/your/certificate1", "/path/to/your/certificate2")


class SentimentServer(Flask):
    def __init__(self, import_name, static_url_path=None, static_folder='static', static_host=None, host_matching=False,
                 subdomain_matching=False, template_folder='templates', instance_path=None,
                 instance_relative_config=False, root_path=None):
        super().__init__(import_name, static_url_path, static_folder, static_host, host_matching, subdomain_matching,
                         template_folder, instance_path, instance_relative_config, root_path)

        self.sentiment = s.SentimentFilter()
        self.prepare_tensorflow()

    def prepare_tensorflow(self):
        print("Preparing weights...")
        print(self.sentiment.get_analysis("Привет!"))

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        @self.route("/", methods=["GET", "POST"])
        def index():
            return jsonify({
                "status": True,
                "message": """Hello, sentiment! <{"result": POSITIVE}>"""
            }), 200

        @self.route("/get_analysis", methods=["GET", "POST"])
        def get_analysis():
            data = request.args.to_dict() or request.form or request.json or request.data or {}

            if "text" not in data:
                return jsonify({
                    "status": False,
                    "error": "Text is not passed in request body."
                }), 500

            scores = None
            if "min_score" in data and "max_score" in data:
                if float(data["min_score"]) < 0:
                    return jsonify({
                        "status": False,
                        "error": "Minimal score cannot be smaller than zero."
                    }), 500

                if float(data["max_score"]) > 1:
                    return jsonify({
                        "status": False,
                        "error": "Maximum score cannot be bigger than one."
                    }), 500

                scores = [float(data["min_score"]), float(data["max_score"])]

            result = self.sentiment.get_analysis(data["text"], scores=scores)

            return jsonify({
                "status": True,
                "response": {
                    "requested_text": data["text"],
                    "score": round(float(result["score"]), 4),
                    "type": result["result"].value
                }
            }), 200

        super().run(host, port, debug, load_dotenv, **options)


ssl_context = None
if USE_SSL:
    ssl_context = SSLContext()
    ssl_context.load_cert_chain(*CERTS)

sentiment_server = SentimentServer(__name__)
sentiment_server.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, ssl_context=ssl_context)
