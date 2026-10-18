import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from scripts.analysis.rev_s_e2_eval_c3 import install_threadlocal_c3


def test_shared_c3_temperature_race_isolated_per_thread():
    started=threading.Event();done=threading.Event()
    class Runner:
        def __init__(self,config,**kwargs):
            self._config=config;self.llm=SimpleNamespace(temperature=config['temperature'])
            self.num_samples=5;self.sample_rows=3;self.desc_dir=None;self.column_descriptions={}
        def generate(self,nl):
            if nl=='second':assert started.wait(2)
            realized=self.llm.temperature
            self.llm.temperature=.7
            if nl=='first':
                started.set();assert done.wait(2)
            else:done.set()
            self.llm.temperature=realized
            return realized
    runner=Runner({'temperature':0})
    def pair():
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(runner.generate,'first')
            second=pool.submit(runner.generate,'second')
            return first.result(),second.result()
    assert pair()==(0,.7)  # Reproduced linking request configuration leak.
    started.clear();done.clear();runner.llm.temperature=0
    install_threadlocal_c3(Runner)
    assert pair()==(0,0)
