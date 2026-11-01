import json
from io import BytesIO
from openpyxl import Workbook
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.main import app, Base, User, dbdep, current_user, hash_password

@pytest.fixture
def api_client():
    test_engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, expire_on_commit=False)
    db = TestSession()
    user = User(username='tester', email='tester@example.com', password=hash_password('test-password-123'))
    db.add(user); db.commit(); db.refresh(user); db.close()
    def override_db():
        session = TestSession()
        try: yield session
        finally: session.close()
    app.dependency_overrides[dbdep] = override_db
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    test_engine.dispose()

def test_dataset_upload_analysis_issue_cleaning_and_reports(api_client):
    csv = b"record_id,email,age,amount,split,target\n1,bad-email,24,10,train,yes\n1,bad-email,24,10,test,yes\n2,valid@example.com,-2,12,train,yes\n3,other@example.com,35,100000,train,no\n"
    uploaded = api_client.post('/api/datasets/upload', files={'file': ('quality.csv', csv, 'text/csv')})
    assert uploaded.status_code == 200
    dataset_id = uploaded.json()['id']
    preview = api_client.get(f'/api/datasets/{dataset_id}/preview')
    assert preview.status_code == 200 and len(preview.json()['rows']) == 4
    analysis = api_client.post(f'/api/analyses/{dataset_id}/run', json={'target_column':'target','split_column':'split'})
    assert analysis.status_code == 200
    result = analysis.json(); analysis_id = result['id']
    assert result['overall_score'] <= 100 and result['readiness']['target_column'] == 'target'
    issues = api_client.get(f'/api/issues?limit=200').json()
    assert any(i['category'] == 'Validity' for i in issues)
    issue_id = issues[0]['id']
    assert api_client.patch(f'/api/issues/{issue_id}/status', json={'status':'Reviewed'}).json()['status'] == 'Reviewed'
    options = {'remove_duplicates':True,'trim_whitespace':True,'remove_empty_columns':True,'missing_strategy':'none','range_rules':{'age':{'min':0,'max':100}}}
    proposed = api_client.post(f'/api/datasets/{dataset_id}/clean/preview', json=options)
    assert proposed.status_code == 200 and proposed.json()['before']['rows'] == 4
    applied = api_client.post(f'/api/datasets/{dataset_id}/clean', json=options)
    assert applied.status_code == 200 and applied.json()['after']['rows'] == 4
    assert 'range:age' in ','.join(applied.json()['operations'])
    assert api_client.get(f'/api/datasets/{dataset_id}/cleaning-history').json()
    assert api_client.get(f'/api/datasets/{dataset_id}/download-cleaned').status_code == 200
    assert api_client.get(f'/api/analyses/{analysis_id}/report/json').status_code == 200
    assert api_client.get(f'/api/analyses/{analysis_id}/report/csv').status_code == 200
    assert api_client.get(f'/api/analyses/{analysis_id}/report/pdf').headers['content-type'] == 'application/pdf'
    assert api_client.get('/api/dashboard/summary').json()['total_datasets'] == 1
    assert api_client.delete(f'/api/datasets/{dataset_id}').status_code == 200
    workbook=Workbook(); workbook.active.title='Default'; workbook.active.append(['value']); workbook.active.append([1]); extra=workbook.create_sheet('Extra'); extra.append(['value']); extra.append([2]); extra.append([3]); payload=BytesIO(); workbook.save(payload)
    files={'file':('sheets.xlsx',payload.getvalue(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    assert api_client.post('/api/datasets/sheets',files=files).json()['sheets']==['Default','Extra']
    selected=api_client.post('/api/datasets/upload',files=files,data={'sheet_name':'Extra'}).json()
    assert selected['sheet_name']=='Extra' and selected['row_count']==2
    assert api_client.delete(f"/api/datasets/{selected['id']}").status_code==200
