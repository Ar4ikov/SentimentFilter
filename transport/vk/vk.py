# | Created by Ar4ikov
# | Время: 16.02.2020 - 22:44

from json import loads
from ssl import SSLContext
from threading import Thread, main_thread

from requests import get, Session
from os import path
from time import time, sleep
from sqlite3 import connect as database

from flask import Flask, request, send_file
import sentiment_filter as s

DEBUG = False
HOST = "0.0.0.0"
# Не трогай порт. С ВК так не прокатит
PORT = 80

USE_POLLING = True

# VK Access token
ACCESS_TOKEN = "token"

# Код подтверждения Callback-сервера
CONFIRMATION_KEY = "your-key"

# Использования секретного кода для Callback-сервера
USE_SECRET = True
SECRET_KEY = "secret_key"
FILE_KEY = "key"

USE_SSL = False
CERTS = ("/path/to/your/certificate1", "/path/to/your/certificate2")


class VkTransport(Flask):
    def __init__(self, import_name, static_url_path=None, static_folder='static', static_host=None, host_matching=False,
                 subdomain_matching=False, template_folder='templates', instance_path=None,
                 instance_relative_config=False, root_path=None):
        super().__init__(import_name, static_url_path, static_folder, static_host, host_matching, subdomain_matching,
                         template_folder, instance_path, instance_relative_config, root_path)

        # SentimentFilter preparing
        self.sentiment = s.SentimentFilter()
        self.prepare_tensorflow()

        # Preparing database for storing comments
        self.table_name_ = "vk_data"
        self.database = database(path.join(path.dirname(__file__), "vk_data.db"), check_same_thread=False)
        self.database_deploy(self.table_name_)

        self.request_session = Session()
        self.headers = loads(open(path.join(path.dirname(__file__), "headers.json"), "r").read())

    def database_deploy(self, table_name):
        self.database.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` 
            (
                `id` INTEGER PRIMARY KEY AUTOINCREMENT ,
                `vk_user` TEXT NOT NULL ,
                `vk_type` TEXT NOT NULL ,
                `vk_object_id` TEXT NOT NULL ,
                `text` TEXT NOT NULL ,
                `vk_group` TEXT NOT NULL ,
                `date` DATE default '01-01-2020' ,
                `estimation` TEXT NOT NULL ,
                `score` DOUBLE NOT NULL 
            );
            """
        )

        self.database.commit()

    def get_method(self, method, vk_version, **data):
        data.update({
            "access_token": ACCESS_TOKEN,
            "v": vk_version
        })

        response = self.request_session.post(f"https://api.vk.com/method/{method}", data=data, headers=self.headers)

        return response.json()

    @staticmethod
    def get_object_info(data):
        vk_text = data["object"]["text"]
        vk_user = data["object"]["from_id"]
        vk_type = data["type"]
        vk_object_id = data["object"]["id"]
        vk_group = data["object"]["owner_id"]
        event_time = data["object"]["date"]

        return vk_text, vk_user, vk_type, vk_object_id, vk_group, event_time

    def long_poll(self):
        class Long_Poll(Thread):
            def __init__(self, root):
                super().__init__()

                self.root = root
                self.name = "LongPoll"

            def update_server_params(self):
                group_id = self.root.get_method("groups.getById", "5.103")["response"][0]["id"]
                response = self.root.get_method("groups.getLongPollServer", "5.103", group_id=group_id)["response"]

                response.update({"act": "a_check", "mode": 2, "version": 2, "wait": 25})

                return response

            def run(self):
                self.params = self.update_server_params()
                self.server = self.params.pop("server")

                print("Long Polling starting...")

                while True:
                    if not main_thread().is_alive():
                        break

                    response = self.root.request_session.post(self.server, data=self.params).json()

                    if "failed" in response:
                        self.params = self.update_server_params()
                        self.server = self.params.pop("server")

                        continue

                    for event in response["updates"]:
                        if event["type"] == "wall_post_new" or event["type"] == "wall_reply_new":
                            vk_text, vk_user, vk_type, vk_object_id, vk_group, event_time = self.root.get_object_info(event)
                            result = self.root.sentiment.get_analysis(vk_text)

                            self.root.database.execute(
                                f"""
                                INSERT INTO {self.root.table_name_} (`vk_user`, `vk_type`, `vk_object_id`, `text`, `vk_group`, `date`, `estimation`, `score`)
                                VALUES ('{vk_user}', '{vk_type}', '{vk_object_id}', '{vk_text}', '{vk_group}', '{event_time}', '{round(float(result["score"]), 4)}', '{result["result"].value}');
                                """
                            )

                            self.root.database.commit()

                    self.params.update({"ts": response["ts"]})

                    sleep(0.001)

        return Long_Poll

    def prepare_tensorflow(self):
        print("Preparing weights...")
        print(self.sentiment.get_analysis("Привет!"))

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        @self.route("/", methods=["GET", "POST"])
        def index():
            return "ok", 200

        @self.route("/get", methods=["GET"])
        def get():
            data = request.json or request.args.to_dict() or request.form or request.data or {}

            if "file_key" not in data or data.get("file_key") != FILE_KEY:
                return "ne ok", 500

            file_path = self.database.execute("PRAGMA database_list;").fetchone()[-1]
            file_name = file_path.split("\\")[-1]

            return send_file(filename_or_fp=file_path, as_attachment=True, attachment_filename=file_name), 200

        if USE_POLLING:
            long_poll = self.long_poll()(self)
            long_poll.start()

        else:
            @self.route("/get_analysis", methods=["GET", "POST"])
            def get_analysis():
                data = request.json or request.args.to_dict() or request.form or request.data or {}
                print(data)

                if data.get("type") == "confirm":
                    return CONFIRMATION_KEY, 200

                if USE_SECRET:
                    if "secret_key" not in data or data.get("secret_key") != SECRET_KEY:
                        return "ne ok", 500

                if data.get("type") == "wall_reply_new" or data["type"] == "wall_post_new":
                    vk_text, vk_user, vk_type, vk_object_id, vk_group, event_time = self.get_object_info(data)
                    response = self.sentiment.get_analysis(vk_text)

                    self.database.execute(
                        f"""
                        INSERT INTO {self.table_name_} (`vk_user`, `vk_type`, `vk_object_id`, `text`, `vk_group`, `date`, `estimation`, `score`)
                        VALUES ('{vk_user}', '{vk_type}', '{vk_object_id}', '{vk_text}', '{vk_group}', '{event_time}', '{round(float(response["score"]), 4)}', '{response["result"].value}');
                        """
                    )

                    self.database.commit()

                return "ok"

        super().run(host, port, debug, load_dotenv, **options)


ssl_context = None
if USE_SSL:
    ssl_context = SSLContext()
    ssl_context.load_cert_chain(*CERTS)

vk_transport = VkTransport(__name__)
vk_transport.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, ssl_context=ssl_context)
