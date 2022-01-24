### 语义搜索API说明


```
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
            'region': ['510107'],
            'deathCount': [0, 10],
            'injuredCount': [112, None],
            'fireReason': [1]
        }
    }
```
