# -*- coding: utf-8 -*-
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models
import base64
import sys
import re

from asrlib.utils import base, reader
from openasr.config import *

if sys.version_info.major == 3:
    raise TypeError('tencent api only support python2')


def to_mp3(wav_stream):
    return base.run_cmd('sox -t wav - -t mp3 -', input_stream=wav_stream, output_stream=base.BytesIO())[0]


def recognize_wav(wav_path, show_detail=True):
    # 通过本地语音上传方式调用
    try:

        # 重要：<Your SecretId>、<Your SecretKey>需要替换成用户自己的账号信息
        # 请参考接口说明中的使用步骤1进行获取。
        cred = credential.Credential(tencent_secret_id, tencent_secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "asr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        clientProfile.signMethod = "TC3-HMAC-SHA256"
        client = asr_client.AsrClient(cred, "ap-shanghai", clientProfile)

        # 读取文件以及base64
        with reader.SaveOpen(to_mp3(wav_path), 'rb') as fwave:
            data = str(fwave.read())
        dataLen = len(data)
        base64Wav = base64.b64encode(data)

        # 发送请求
        req = models.SentenceRecognitionRequest()
        params = {"ProjectId": 0, "SubServiceType": 2, "EngSerViceType": "16k_en", "SourceType": 1, "Url": "",
                  "VoiceFormat": "mp3", "UsrAudioKey": "session-123", "Data": base64Wav, "DataLen": dataLen}
        req._deserialize(params)
        resp = client.SentenceRecognition(req)
        if show_detail:
            print(resp.to_json_string())
        # windows系统使用下面一行替换上面一行
        # print(resp.to_json_string().decode('UTF-8').encode('GBK') )

        words = []

        for w in resp.Result.split():
            if not re.match('[a-zA-W]', w[-1:]):
                w = w[0:-1]
            words.append(w)
        return ' '.join(words).lower()

    except TencentCloudSDKException as err:
        print(err)
        return '[ERROR]'


if __name__ == '__main__':
    print(recognize_wav('../testdata/a.wav'))
