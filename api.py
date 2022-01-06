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
"""
sequence labeling
"""
import ast
import os
import json
import warnings
import random
import argparse
import time
from functools import partial

import numpy as np
import paddle
import jionlp as jio
import utils
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from flask import Flask, request, jsonify
from utils import load_dict, read_by_lines, extract_result

warnings.filterwarnings('ignore')
app = Flask(__name__)


def convert_example_to_feature(example, tokenizer, label_vocab=None, max_seq_len=512, no_entity_label="O",
                               ignore_label=-1, is_test=False):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len - 2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        return input_ids, token_type_ids, seq_len, encoded_label


def do_predict(args, text=''):
    """
    Do predict for single sentence
    Either trigger or role are ok.
    """

    paddle.set_device(args.device)

    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_map))

    no_entity_label = "O"
    ignore_label = len(label_map)

    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        state_dict = paddle.load(args.init_ckpt)
        model.set_dict(state_dict)

    # load data from predict file
    sentences = [{'text': text, 'id': 'a7c74f75eb8986377096b4dc62db217d'}]  # origin data format

    encoded_inputs_list = []
    for sent in sentences:
        sent = sent["text"].replace(" ", "\002")
        input_ids, token_type_ids, seq_len = convert_example_to_feature([list(sent), []], tokenizer,
                                                                        max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'),  # token_type_ids
        Stack(dtype='int64')  # sequence lens
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [encoded_inputs_list[i: i + args.batch_size]
                            for i in range(0, len(encoded_inputs_list), args.batch_size)]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=-1)
        probs_ids = paddle.argmax(probs, -1).numpy()
        probs = probs.numpy()
        for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(), seq_lens.tolist()):
            prob_one = [p_list[index][pid] for index, pid in enumerate(p_ids[1: seq_len - 1])]
            label_one = [id2label[pid] for pid in p_ids[1: seq_len - 1]]
            results.append({"probs": prob_one, "labels": label_one})
    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    return sentences


def predict_data_process(trigger_data, role_data, schema_file):
    """
    Predict data process for single sentence
    """

    pred_ret = []
    schema_datas = read_by_lines(schema_file)

    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]

    sent_role_mapping = {}
    for d in role_data:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append("".join(r["text"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_data:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        for event_type in pred_event_types:
            if event_type not in schema:
                continue
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    return pred_ret


def combine_predict(text):
    """
    Combine from do_predict to predict_data_process for trigger and role
    """

    parser = argparse.ArgumentParser(__doc__, add_help=False)
    utils.load_yaml(parser, './conf/args.yaml')
    args = parser.parse_args()

    trigger_parser = argparse.ArgumentParser(parents=[parser])
    utils.load_yaml(trigger_parser, './conf/trigger_args.yaml')
    trigger_args = trigger_parser.parse_args()

    role_parser = argparse.ArgumentParser(parents=[parser])
    utils.load_yaml(role_parser, './conf/role_args.yaml')
    role_args = role_parser.parse_args()

    trigger_data = do_predict(trigger_args, text=text)
    print(trigger_data)
    role_data = do_predict(role_args, text=text)
    print(role_data)
    comb_data = predict_data_process(trigger_data, role_data, args.schema_file)
    print(comb_data)


def role_predict(text):
    """
    Predict only role for single sentence
    """

    parser = argparse.ArgumentParser(__doc__, add_help=False)
    utils.load_yaml(parser, './conf/args.yaml')

    role_parser = argparse.ArgumentParser(parents=[parser])
    utils.load_yaml(role_parser, './conf/role_args.yaml')
    role_args = role_parser.parse_args()

    role_data = do_predict(role_args, text=text)

    sent_role_mapping = {}
    for d in role_data:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []

            role_text = "".join(r["text"])
            if role_type == '时间':
                acc_role = jio.parse_time(role_text, time.time())
                role_text = acc_role['time']
            elif role_type == '地点':
                acc_role = jio.parse_location(role_text)
                role_text = acc_role

            role_ret[role_type] = role_text
        sent_role_mapping[d_json["id"]] = role_ret
    return sent_role_mapping


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
    return jsonify(role_predict(**data))


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=5000, debug=True)
    result = role_predict('昨天成都市武侯区，火灾共导致85人死亡，112人受伤')
    print(result)
    result = role_predict('2021年12月 绵竹')
    print(result)
    result = role_predict('化工厂 1人死亡')
    print(result)
    result = role_predict('绵阳市 7天酒店')
    print(result)
    result = role_predict('阿坝州火灾')
    print(result)
