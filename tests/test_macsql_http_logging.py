import json
import importlib
import openai
httpx = importlib.import_module(openai.DefaultHttpxClient.__mro__[1].__module__.split('.')[0])
from egrefine.models.llm_client import LLMClient


def test_raw_http_log_preserves_failure_and_sdk_retry_before_parsing(tmp_path, monkeypatch):
    path=tmp_path/'raw.jsonl';monkeypatch.setenv('EGREFINE_HTTP_RAW_LOG',str(path))
    n=0
    def respond(request):
        nonlocal n
        n+=1
        if n==1:
            return httpx.Response(500,json={'error':{'message':'original raw failure','type':'server_error'}},request=request)
        return httpx.Response(200,json={'id':'x','object':'chat.completion','created':1,'model':'test','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'RAW answer'}}]},request=request)
    client=LLMClient({'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy'})
    client._client._client._transport=httpx.MockTransport(respond)
    assert client.chat([{'role':'user','content':'Complete input'}])=='RAW answer'
    events=[json.loads(line) for line in path.read_text().splitlines()]
    assert [e['event'] for e in events]==['request','response','request','response']
    assert [e['status_code'] for e in events if e['event']=='response']==[500,200]
    assert 'original raw failure' in events[1]['body']
    assert 'Complete input' in events[0]['body'] and events[0]['body']==events[2]['body']
    assert events[0]['id']==events[1]['id'] and events[2]['id']==events[3]['id']
