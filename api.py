import warnings

from flask import Flask, request, jsonify
from python.predict import role_infer
from predict import role_predict

warnings.filterwarnings('ignore')
app = Flask(__name__)


@app.route("/keyword", methods=["GET", "POST"])
def info_test():
    """
    返回标签和实体，支持 post 和 get 请求

    kwargs:
        text: 搜索文本

    post:
        url: http://127.0.0.1:5000/keyword
        kwargs: {"text": ...}
    get:
        url: http://127.0.0.1:5000/keyword?text...
    """
    data = request.form if request.method == "POST" else request.args
    return jsonify(role_infer(**data))


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=5000, debug=True)
    print(role_infer('1月4日12点 武侯 0-十人死亡 112人以上受伤 起火原因：电气短路'))
