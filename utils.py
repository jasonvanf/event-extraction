# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import re
import yaml
import json
import time

import cn2an
from difflib import SequenceMatcher

import jionlp as jio

NAME_CODE = {
    '行政区域': {'code': 'region', 'group': 4},
    '单位类型': {
        'code': 'deptType',
        'group': 0,
        'option': {
            '所有类型': 0,
            '重点单位': 1,
            '一般单位': 2,
            '九小场所': 3,
            '其他单位': 9,
            'default': 9,
        },
    },
    '隐患数量': {'code': 'hiddenCount', 'group': 1},
    '火灾数量': {'code': 'fireCount', 'group': 1},
    '火灾损失': {'code': 'fireLoss', 'group': 1},
    '建筑类型': {
        'code': 'buildType',
        'group': 0,
        'option': {
            '全部类型': 0,
            '高层': 10000,
            '多层': 20000,
            '单层': 30000,
            '地下': 40000,
            'default': 0,
        },
    },
    '建筑面积': {'code': 'buildArea', 'group': 1},
    '死亡人数': {'code': 'deathCount', 'group': 1},
    '受伤人数': {'code': 'injuredCount', 'group': 1},
    '报警时间': {'code': 'alarmTime', 'group': 3},
    '过火面积': {'code': 'fireArea', 'group': 1},
    '起火原因': {
        'code': 'fireReason',
        'group': 0,
        'option': {
            '全部原因': 0,
            '电气火灾': 1,
            '生产作业类火灾': 2,
            '生活用火不慎': 3,
            '吸烟': 4,
            '玩火': 5,
            '自燃': 6,
            '雷击': 7,
            '静电': 8,
            '不明确原因': 9,
            '放火': 10,
            '其他': 99,
            'default': 99,
        },
    },
}


def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()


def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w") as outfile:
        [outfile.write(d + "\n") for d in data]


def text_to_sents(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    paragraphs = text.split("\n")
    ret = []
    for para in paragraphs:
        if para == u"":
            continue
        sents = [u""]
        for s in para:
            sents[-1] += s
            if s in deliniter_symbols:
                sents.append(u"")
        if sents[-1] == u"":
            sents = sents[:-1]
        ret.extend(sents)
    return ret


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    if len(text) != len(labels):
        # 韩文回导致label 比 text要长
        labels = labels[:len(text)]
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


def predict_group_0(name, text):
    options = NAME_CODE[name]['option']

    if not isinstance(options, dict):
        return [0]

    text = "".join(text)
    max_ratio = 0.0
    max_ratio_key = options['default']
    for key, val in options.items():
        ratio = SequenceMatcher(None, key, text).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            max_ratio_key = val

    if max_ratio >= 0.5:
        result = [max_ratio_key]
    else:
        result = [options['default']]
    return result


def predict_group_1(name, ori_text):
    text_list = re.split('-|到', ori_text)

    if len(text_list) > 2:
        return [None, None]

    for idx, text in enumerate(text_list):
        text_list[idx] = "".join([s for s in list(text) if s.isnumeric()])
        text_list[idx] = int(cn2an.cn2an(text_list[idx], 'smart'))

    if len(text_list) != 2:
        text_list.append(text_list[0])

    if '以下' in ori_text:
        text_list[0] = None
    elif '以上' in ori_text:
        text_list[1] = None

    return text_list


def predict_group_3(name, text):
    acc_role = jio.parse_time(text, time_base=time.time(), time_type='time_span')
    return acc_role['time']


def predict_group_4(name, text):
    path = ['5101']
    with open('./conf/doccano/region.json', 'r', encoding='utf-8') as fp:
        d_json = json.loads(fp.read())

        acc_role = jio.parse_location(text)
        if 'county' in acc_role:
            path = [id for id, name in d_json.items() if name == acc_role['county']]

        if 'city' in acc_role and len(path) == 0:
            path = [id for id, name in d_json.items() if name == acc_role['city']]

    return path


def predict_group(r_group, name, text):
    return {
        0: predict_group_0,
        1: predict_group_1,
        3: predict_group_3,
        4: predict_group_4,
    }.get(r_group)(name, text)


def predict2json(data):
    sent_role_mapping = {}
    for d in data:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = NAME_CODE[r["type"]]['code']
            role_group = NAME_CODE[r["type"]]['group']
            if role_type not in role_ret:
                role_ret[role_type] = []

            role_text = "".join(r["text"])
            role_res = predict_group(role_group, r["type"], role_text)

            role_ret[role_type] = role_res
        sent_role_mapping[d_json["id"]] = role_ret
    return sent_role_mapping


def str2bool(v):
    """
    argparse does not support True or False in python
    """
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Put arguments to one group
    """

    def __init__(self, parser, title, des):
        """none"""
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """ Add argument """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def load_yaml(parser, file_name, **kwargs):
    with open(file_name, 'r', encoding='utf8') as f:
        args = yaml.safe_load(f)
        for title in args:
            group = parser.add_argument_group(title=title, description='')
            for name in args[title]:
                _type = type(args[title][name]['val'])
                _type = str2bool if _type == bool else _type
                group.add_argument(
                    "--" + name,
                    default=args[title][name]['val'],
                    type=_type,
                    help=args[title][name]['meaning'] + ' Default: %(default)s.',
                    **kwargs)


def print_arguments(args):
    """none"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    s = "xxdedewd"
    print(cal_md5(s.encode("utf-8")))
