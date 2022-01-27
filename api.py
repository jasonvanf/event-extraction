import warnings

from flask import Flask, request, jsonify
from python.predict import role_infer
from predict import role_predict

warnings.filterwarnings('ignore')
app = Flask(__name__)


@app.route("/keyword", methods=["GET", "POST"])
def info_test():
    """
    url: http://172.19.164.120:5000/keyword
    返回标签和实体，支持 post 请求

    kwargs:
        text: 搜索文本

    post:
        url: http://127.0.0.1:5000/keyword
        kwargs: "text": ...

    return:
        {0:
            {
                'alarmTime': ['2022-01-04 12:00:00', '2022-01-04 12:59:59'],
                'region': {'province': '四川省', 'city': '成都市', 'county': '武侯区', 'detail': '', 'full_location': '四川省成都市武侯区', 'orig_location': '武侯'},
                'deathCount': [0, 10],
                'injuredCount': [112, None],
                'fireReason': [1]
            }
        }
    """
    data = request.form if request.method == "POST" else request.args

    try:
        sent_role_mapping = role_infer(**data)
        result = '' if '0' in sent_role_mapping and len(sent_role_mapping['0']) < 1 else jsonify(sent_role_mapping)
    except Exception:
        result = ''

    return result


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    # text = [
    #     '1月4日12点 武侯 0-十人死亡 112人以上受伤 起火原因：电气短路',
    #     '静电 高层',
    # ]
    # print(role_infer(text))
