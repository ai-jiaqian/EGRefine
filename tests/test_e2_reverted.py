import json
import pytest
from scripts.analysis.build_e2_reverted import build


def test_multiple_reverts_in_one_database_are_retained(tmp_path):
    source=tmp_path/'source'/'db';source.mkdir(parents=True)
    (source/'views.sql').write_text('CREATE VIEW "t" AS\nSELECT\n  "a" AS "alpha",\n  "b" AS "beta"\nFROM "_orig_t";')
    (source/'refined_tables.json').write_text(json.dumps({'tables':[{'name':'t','columns':[
        {'name':'alpha','original_name':'a'},{'name':'beta','original_name':'b'}]}],
        'foreign_keys':[['u.x','t.alpha']]}))
    (source/'pipeline_result.json').write_text(json.dumps({'refinements':[
        {'table':'t','original_name':a,'refined_name':b,'was_changed':True} for a,b in [('a','alpha'),('b','beta')]],
        'mapping':{'t.a':'alpha','t.b':'beta'}}))
    output=tmp_path/'output'
    items=[('db','t','a','alpha'),('db','t','b','beta')]
    build(source.parent,output,items)
    result=json.loads((output/'db'/'refined_tables.json').read_text())
    assert [c['name'] for c in result['tables'][0]['columns']]==['a','b']
    assert result['foreign_keys']==[['u.x','t.a']]
    assert ' AS "alpha"' not in (output/'db'/'views.sql').read_text()
    assert ' AS "beta"' not in (output/'db'/'views.sql').read_text()
    with pytest.raises(FileExistsError):
        build(source.parent,output,items)
