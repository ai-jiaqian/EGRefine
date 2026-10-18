import hashlib
import inspect
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.macsql_runner import MACSQLText2SQL
from egrefine.phase3.macsql_official_runner import MACSQLOfficialText2SQL, MAX_ROUND
from egrefine.phase3 import macsql_official_prompts as prompts
from egrefine.phase3.macsql_official_utils import parse_sql_from_string, parse_json

REF = Path(__file__).resolve().parent / 'fixtures/macsql_prompt_hashes.json'


def runner(tmp_path, dataset='spider'):
    return MACSQLOfficialText2SQL({'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy','trace_dir':str(tmp_path)}, dataset_name=dataset)


def schema(counts):
    return Schema('toy', [Table(f't{i}', [Column(f'c{j}', f't{i}', 'TEXT') for j in range(n)]) for i,n in enumerate(counts)], [])


def test_signature():
    assert inspect.signature(MACSQLOfficialText2SQL.generate) == inspect.signature(MACSQLText2SQL.generate)


@pytest.mark.parametrize('name', ['selector_template','decompose_template_spider','decompose_template_bird','refiner_template'])
def test_verbatim_prompts(name):
    reference = json.loads(REF.read_text())['templates'][name]
    assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == reference['raw_sha256']
    values=dict(db_id='toy',query='QUESTION',evidence='EVIDENCE',desc_str='SCHEMA',fk_str='FK',sql='SELECT 1',sqlite_error='ERROR',exception_class='CLASS')
    rendered = getattr(prompts, name).format(**values)
    assert hashlib.sha256(rendered.encode()).hexdigest() == reference['rendered_sha256']


@pytest.mark.parametrize('text,sql', [('```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```','SELECT 2'),('SELECT 1','error: No SQL found in the input string'),('```SQL\nSELECT 1\n```','error: No SQL found in the input string')])
def test_sql_extraction(text,sql):
    assert parse_sql_from_string(text)==sql


@pytest.mark.parametrize('counts,expected', [([6]*5,False),([6]*6,True),([7],True),([7,6],False),([0,6],False)])
def test_threshold(counts,expected):
    assert MACSQLOfficialText2SQL._is_need_prune(schema(counts))==expected


@pytest.mark.parametrize('dataset,result,need',[('spider',{'data':[]},False),('spider',{'data':[(None,)]},False),('spider',{'sqlite_error':'bad'},True),('bird',{'data':[]},True),('bird',{'data':[(None,)]},True),('bird',{'data':[(1,)]},False),('bird',{'sqlite_error':'bad'},True)])
def test_refiner_trigger(tmp_path,dataset,result,need):
    assert runner(tmp_path,dataset)._is_need_refine(result)==need


@pytest.mark.parametrize('counts,agents',[([3],['Decomposer']),([7],['Selector','Decomposer'])])
def test_selector_branch_and_raw_log(tmp_path,counts,agents):
    m=runner(tmp_path);m.llm.chat=MagicMock(side_effect=['```json\n{"t0":"keep_all"}\n```','```sql\nSELECT 1\n```'] if len(agents)==2 else ['```sql\nSELECT 1\n```'])
    assert m.generate('q',schema(counts))=='SELECT 1'
    r=json.loads(next(tmp_path.rglob('macsql_calls/*.json')).read_text())
    assert [c['agent'] for c in r['calls']]==agents
    assert r['calls'][-1]['raw_response']=='```sql\nSELECT 1\n```'
    assert r['calls'][-1]['messages'][0]['content']==prompts.decompose_template_spider.format(desc_str=r['desc_str'],fk_str=r['fk_str'],query='q')


def test_three_refinements_even_when_sql_unchanged(tmp_path):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE t0 (c0 TEXT, c1 TEXT)')
    m=runner(tmp_path);m.llm.chat=MagicMock(return_value='```sql\nSELECT missing FROM absent\n```')
    m.generate('q',schema([2]),str(db))
    assert m.llm.chat.call_count==1+MAX_ROUND


def test_refined_values_and_execution(tmp_path):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as c:
        c.executescript("CREATE TABLE t0 (nm TEXT); INSERT INTO t0 VALUES ('Alice'),('Alice'),('Bob');")
    s=Schema('toy',[Table('t0',[Column('person_name','t0','TEXT')])],[])
    m=runner(tmp_path)
    desc,_,_=m._get_db_desc_str('toy',m._schema_info(s,str(db),{'person_name':'nm'}),{})
    assert desc=="# Table: t0\n[\n  (person_name, person name. Value examples: ['Alice', 'Bob'].)\n]"
    assert m._execute_sql('SELECT person_name FROM t0',str(db),{'person_name':'nm'})['data']==[('Alice',),('Alice',),('Bob',)]


def test_bird_description_uses_visible_names_only(tmp_path):
    d=tmp_path/'toy/database_description';d.mkdir(parents=True)
    (d/'t0.csv').write_text('original_column_name,column_description,value_description\nnm,Secret original meaning,unit\n')
    m=MACSQLOfficialText2SQL({'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy','trace_dir':str(tmp_path)},dataset_name='bird',desc_dir=str(tmp_path))
    s=Schema('toy',[Table('t0',[Column('person_name','t0','TEXT')])],[])
    info=m._schema_info(s,None,{'person_name':'nm'})
    assert info['desc_dict']['t0'][0][2]==''


def test_description_baseline_injects_same_visible_column_text_as_din(tmp_path):
    from egrefine.phase3.dinsql_runner import DINSQLText2SQL
    config={'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy','trace_dir':str(tmp_path)}
    s=schema([2]);description='Full meaning ' + 'long ' * 30 + '*/\nend'
    descriptions={('toy','t0','c0'):description, ('toy','t0','hidden_old_name'):'SECRET'}
    m=MACSQLOfficialText2SQL(config,column_descriptions=descriptions)
    plain=runner(tmp_path)
    desc=m._get_db_desc_str('toy',m._schema_info(s,None,{}),{})[0]
    din=DINSQLText2SQL(config,column_descriptions=descriptions)._format_ddl(s)
    annotation='/* '+description.replace('*/',' ').replace('\n',' ')+' */'
    assert annotation in desc and annotation in din
    assert 'SECRET' not in desc
    # No descriptions keeps the official schema renderer byte-for-byte unchanged.
    assert MACSQLOfficialText2SQL(config,column_descriptions={})._schema_info(s,None,{}) == plain._schema_info(s,None,{})


def test_selector_official_drop_and_key_padding(tmp_path):
    m=runner(tmp_path); s=schema([8]);s.tables[0].columns[7].is_pk=True
    info=m._schema_info(s,None,{})
    _,_,chosen=m._get_db_desc_str('toy',info,{'t0':['c6']})
    assert chosen['t0']==['c6','c7','c0','c1','c2','c3']
    _,_,chosen=m._get_db_desc_str('toy',info,{'t0':'drop_all'})
    assert chosen['t0']==['c0','c1','c2','c3','c4','c5']
    assert parse_json('{"t0":"keep_all"}')=={}


def test_metadata_scope_and_threshold(tmp_path):
    metadata=[{'db_id':'toy','table_names_original':['t0'],
               'column_names_original':[[-1,'*'],[0,'c0']],
               'column_names':[[-1,'*'],[0,'c0']], 'foreign_keys':[], 'primary_keys':[]}]
    path=tmp_path/'tables.json';path.write_text(json.dumps(metadata))
    m=MACSQLOfficialText2SQL({'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy','trace_dir':str(tmp_path)},tables_json=str(path))
    s=schema([1,40]);s.foreign_keys=[('t1.c0','t0.c0')]
    info=m._schema_info(s,None,{})
    assert list(info['desc_dict'])==['t0']
    assert not m._is_need_prune(s,info)
    assert m._get_db_desc_str('toy',info,{})[1]==''


def test_full_api_trace(tmp_path):
    m=runner(tmp_path)
    completion=MagicMock()
    completion.model_dump.return_value={'choices':[{'finish_reason':'stop','message':{'content':'```sql\nSELECT 1\n```','reasoning':None}}], 'usage':{'total_tokens':42}}
    completion.choices[0].message.content='```sql\nSELECT 1\n```'
    m.llm._client.chat.completions.create=MagicMock(return_value=completion)
    m.generate('q',schema([2]))
    trace=json.loads(next(tmp_path.rglob('macsql_calls/*.json')).read_text())['calls'][0]['api_calls'][0]
    assert trace['request']['temperature']==0
    assert trace['request']['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert trace['response']==completion.model_dump.return_value


def test_official_primary_key_metadata_controls_examples(tmp_path):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE t0 (c0 INT, c1 INT, PRIMARY KEY(c0,c1)); INSERT INTO t0 VALUES (1,7);')
    metadata=[{'db_id':'toy','table_names_original':['t0'], 'column_names_original':[[-1,'*'],[0,'c0'],[0,'c1']], 'column_names':[[-1,'*'],[0,'c0'],[0,'c1']], 'foreign_keys':[], 'primary_keys':[1]}]
    path=tmp_path/'tables.json';path.write_text(json.dumps(metadata))
    m=MACSQLOfficialText2SQL({'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy','trace_dir':str(tmp_path)},tables_json=str(path))
    s=schema([2]);s.tables[0].columns[0].is_pk=True;s.tables[0].columns[1].is_pk=True
    info=m._schema_info(s,str(db),{})
    assert info['value_dict']['t0']==[['c0',''],['c1','[7]']]
    assert info['pk_dict']['t0']==['c0']
    s.tables[0].columns[1].name='refined_grade'
    refined=m._schema_info(s,str(db),{'refined_grade':'c1'})
    assert refined['value_dict']['t0']==[['c0',''],['refined_grade','[7]']]


def test_dispatch():
    from scripts.run_eval import METHOD_REGISTRY
    assert METHOD_REGISTRY['macsql'] is MACSQLOfficialText2SQL
    assert METHOD_REGISTRY['macsql_legacy'] is MACSQLText2SQL


def test_declared_sqlite_sequence_restored(tmp_path):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE t0 (c0 INTEGER PRIMARY KEY AUTOINCREMENT)')
    metadata=[{'db_id':'toy','table_names_original':['t0','sqlite_sequence'], 'column_names_original':[[-1,'*'],[0,'c0'],[1,'name'],[1,'seq']], 'column_names':[[-1,'*'],[0,'c0'],[1,'name'],[1,'seq']], 'foreign_keys':[], 'primary_keys':[1]}]
    path=tmp_path/'tables.json';path.write_text(json.dumps(metadata))
    m=MACSQLOfficialText2SQL({'model_name':'test','base_url':'http://localhost:1/v1','api_key':'dummy','trace_dir':str(tmp_path)},tables_json=str(path))
    info=m._schema_info(schema([1]),str(db),{})
    assert list(info['desc_dict'])==['t0','sqlite_sequence']
    assert [c[0] for c in info['desc_dict']['sqlite_sequence']]==['name','seq']


def test_phase3_refined_fk_uses_visible_identifier(tmp_path):
    db=tmp_path/'toy.sqlite'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE t0 (pid INT); INSERT INTO t0 VALUES (7); CREATE TABLE t1 (id INT PRIMARY KEY); INSERT INTO t1 VALUES (7);')
    original=Schema('toy',[Table('t0',[Column('pid','t0','INT')]),Table('t1',[Column('id','t1','INT',is_pk=True)])],[('t0.pid','t1.id')])
    refined=original.apply_refinement({'t0.pid':'parent_id'})
    m=runner(tmp_path)
    info=m._schema_info(refined,str(db),{'parent_id':'pid'})
    desc,fk,_=m._get_db_desc_str('toy',info,{})
    assert '(parent_id, parent id.)' in desc
    assert fk=='t0.`parent_id` = t1.`id`'
    assert info['value_dict']['t0']==[['parent_id','']]
