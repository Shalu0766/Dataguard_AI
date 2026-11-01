from __future__ import annotations
import io, json, os, re, secrets, shutil, base64, hashlib, hmac
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Text, inspect, text as sql_text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

ROOT=Path(__file__).resolve().parents[1]; UPLOADS=ROOT/'uploads'; UPLOADS.mkdir(exist_ok=True)
DB_URL=os.getenv('DATABASE_URL',f"sqlite:///{ROOT/'dataguard.db'}")
engine=create_engine(DB_URL,connect_args={'check_same_thread':False} if DB_URL.startswith('sqlite') else {})
SessionLocal=sessionmaker(bind=engine,expire_on_commit=False); Base=declarative_base()
SECRET=os.getenv('DATAGUARD_SECRET','development-secret-change-me')
def utcnow(): return datetime.now(timezone.utc).replace(tzinfo=None)
def hash_password(password:str)->str:
 salt=secrets.token_bytes(16)
 digest=hashlib.scrypt(password.encode('utf-8'),salt=salt,n=2**14,r=8,p=1)
 return 'scrypt$'+base64.urlsafe_b64encode(salt).decode()+'$'+base64.urlsafe_b64encode(digest).decode()
def verify_password(password:str,encoded:str)->bool:
 try:
  scheme,salt_text,digest_text=encoded.split('$',2)
  if scheme!='scrypt': return False
  salt=base64.urlsafe_b64decode(salt_text); expected=base64.urlsafe_b64decode(digest_text)
  actual=hashlib.scrypt(password.encode('utf-8'),salt=salt,n=2**14,r=8,p=1)
  return hmac.compare_digest(actual,expected)
 except (ValueError,TypeError): return False
class User(Base):
 __tablename__='users'; id=Column(Integer,primary_key=True); username=Column(String,unique=True,index=True); email=Column(String,unique=True,index=True); password=Column(String); created_at=Column(DateTime,default=utcnow)
class Dataset(Base):
 __tablename__='datasets'; id=Column(Integer,primary_key=True); user_id=Column(Integer,ForeignKey('users.id'),index=True); filename=Column(String); path=Column(String); file_type=Column(String); file_size=Column(Integer); rows=Column(Integer); columns=Column(Integer); uploaded_at=Column(DateTime,default=utcnow); cleaned_path=Column(String,nullable=True); sheet_name=Column(String,nullable=True)
class Analysis(Base):
 __tablename__='analyses'; id=Column(Integer,primary_key=True); dataset_id=Column(Integer,ForeignKey('datasets.id'),index=True); score=Column(Float); category=Column(String); result=Column(Text); created_at=Column(DateTime,default=utcnow)
class Issue(Base):
 __tablename__='issues'; id=Column(Integer,primary_key=True); analysis_id=Column(Integer,ForeignKey('analyses.id'),index=True); category=Column(String); column_name=Column(String,nullable=True); description=Column(String); severity=Column(String); affected_count=Column(Integer); examples=Column(Text); recommendation=Column(String); status=Column(String,default='Open')
class CleaningOperation(Base):
 __tablename__='cleaning_operations'; id=Column(Integer,primary_key=True); dataset_id=Column(Integer,ForeignKey('datasets.id'),index=True); operation_details=Column(Text); before_stats=Column(Text); after_stats=Column(Text); created_at=Column(DateTime,default=utcnow)
Base.metadata.create_all(engine)
if 'sheet_name' not in {c['name'] for c in inspect(engine).get_columns('datasets')}:
 with engine.begin() as conn: conn.execute(sql_text('ALTER TABLE datasets ADD COLUMN sheet_name VARCHAR'))
def dbdep():
 db=SessionLocal()
 try: yield db
 finally: db.close()
def current_user(authorization:str|None=Header(None),db:Session=Depends(dbdep)):
 if not authorization or not authorization.startswith('Bearer '): raise HTTPException(401,'Authentication required')
 try: uid=jwt.decode(authorization[7:],SECRET,algorithms=['HS256'])['sub']
 except (JWTError,KeyError): raise HTTPException(401,'Invalid session')
 user=db.get(User,int(uid))
 if not user: raise HTTPException(401,'Invalid session')
 return user
def owned_dataset(dataset_id:int,user:User,db:Session):
 d=db.query(Dataset).filter_by(id=dataset_id,user_id=user.id).first()
 if not d: raise HTTPException(404,'Dataset not found')
 return d
def load_df(d:Dataset,cleaned=False):
 p=d.cleaned_path if cleaned and d.cleaned_path else d.path
 try:
  if d.file_type=='csv': return pd.read_csv(p)
  return pd.read_excel(p,sheet_name=d.sheet_name or 0)
 except Exception as e: raise HTTPException(422,'Could not read dataset file') from e
app=FastAPI(title='DataGuard AI',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
class Register(BaseModel): username:str; email:EmailStr; password:str
class Login(BaseModel): email:EmailStr; password:str
class ProfileUpdate(BaseModel): username:str|None=None; email:EmailStr|None=None; current_password:str; new_password:str|None=None
class StatusUpdate(BaseModel): status:str
class AnalysisRequest(BaseModel): target_column:str|None=None; split_column:str|None=None; outlier_method:Literal['iqr','zscore','both']='both'; iqr_multiplier:float=Field(default=1.5,gt=0,le=5); zscore_threshold:float=Field(default=3.0,gt=0,le=10); weights:dict[str,float]|None=None
class CleanRequest(BaseModel): remove_duplicates:bool=False; trim_whitespace:bool=False; remove_empty_columns:bool=False; text_case:Literal['none','lower','upper','title']='none'; missing_strategy:str='none'; constant_value:str=''; range_rules:dict[str,dict[str,float]]=Field(default_factory=dict)
@app.get('/api/health')
def health(): return {'status':'ok'}
@app.get('/api/samples')
def samples():
 folder=ROOT.parent/'sample_data'
 return [{'filename':p.name,'size':p.stat().st_size} for p in sorted(folder.glob('*.csv'))]
@app.get('/api/samples/{filename}')
def sample_download(filename:str):
 if Path(filename).name!=filename or not filename.lower().endswith('.csv'): raise HTTPException(404,'Sample not found')
 path=ROOT.parent/'sample_data'/filename
 if not path.is_file(): raise HTTPException(404,'Sample not found')
 return FileResponse(path,media_type='text/csv',filename=filename)
@app.post('/api/auth/register')
def register(body:Register,db:Session=Depends(dbdep)):
 if len(body.password)<8: raise HTTPException(400,'Password must contain at least 8 characters')
 if db.query(User).filter((User.email==body.email)|(User.username==body.username)).first(): raise HTTPException(409,'Email or username is already registered')
 u=User(username=body.username,email=body.email,password=hash_password(body.password)); db.add(u); db.commit(); db.refresh(u)
 token=jwt.encode({'sub':str(u.id),'exp':datetime.now(timezone.utc)+timedelta(days=7)},SECRET,algorithm='HS256')
 return {'token':token,'user':{'id':u.id,'username':u.username,'email':u.email}}
@app.post('/api/auth/login')
def login(body:Login,db:Session=Depends(dbdep)):
 u=db.query(User).filter_by(email=body.email).first()
 if not u or not verify_password(body.password,u.password): raise HTTPException(401,'Email or password is incorrect')
 token=jwt.encode({'sub':str(u.id),'exp':datetime.now(timezone.utc)+timedelta(days=7)},SECRET,algorithm='HS256')
 return {'token':token,'user':{'id':u.id,'username':u.username,'email':u.email}}
@app.get('/api/auth/me')
def me(u:User=Depends(current_user)): return {'id':u.id,'username':u.username,'email':u.email}
@app.patch('/api/auth/profile')
def update_profile(body:ProfileUpdate,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 if not verify_password(body.current_password,u.password): raise HTTPException(401,'Current password is incorrect')
 username=(body.username or '').strip() or u.username; email=body.email or u.email
 conflict=db.query(User).filter(User.id!=u.id,((User.username==username)|(User.email==email))).first()
 if conflict: raise HTTPException(409,'Username or email is already in use')
 if body.new_password and len(body.new_password)<8: raise HTTPException(400,'New password must contain at least 8 characters')
 u.username=username; u.email=str(email)
 if body.new_password: u.password=hash_password(body.new_password)
 db.commit(); return {'id':u.id,'username':u.username,'email':u.email}
@app.post('/api/datasets/upload')
async def upload(file:UploadFile=File(...),sheet_name:str|None=Form(None),u:User=Depends(current_user),db:Session=Depends(dbdep)):
 ext=Path(file.filename or '').suffix.lower()
 if ext not in ('.csv','.xlsx','.xls'): raise HTTPException(415,'Upload a CSV, XLSX, or XLS file')
 data=await file.read()
 if not data: raise HTTPException(400,'The uploaded file is empty')
 if len(data)>50*1024*1024: raise HTTPException(413,'Maximum file size is 50 MB')
 path=UPLOADS/f'{secrets.token_hex(16)}{ext}'; path.write_bytes(data)
 try:
  frame=pd.read_csv(io.BytesIO(data),nrows=250001) if ext=='.csv' else pd.read_excel(io.BytesIO(data),sheet_name=sheet_name or 0,nrows=250001)
  if isinstance(frame,dict): frame=next(iter(frame.values()))
 except Exception:
  path.unlink(missing_ok=True); raise HTTPException(422,'File could not be parsed. Check its format and selected sheet.')
 if len(frame)>250000: path.unlink(missing_ok=True); raise HTTPException(413,'Maximum supported dataset size is 250,000 rows')
 if len(frame.columns)==0: path.unlink(missing_ok=True); raise HTTPException(422,'Dataset has no columns')
 safe_name=re.sub(r'[^A-Za-z0-9._ -]','_',Path(file.filename or 'dataset').name).strip(' .') or 'dataset'+ext
 d=Dataset(user_id=u.id,filename=safe_name,path=str(path),file_type='csv' if ext=='.csv' else 'excel',file_size=len(data),rows=len(frame),columns=len(frame.columns),sheet_name=sheet_name if ext!='.csv' else None); db.add(d); db.commit(); db.refresh(d)
 return dataset_view(d,frame)
@app.post('/api/datasets/sheets')
async def workbook_sheets(file:UploadFile=File(...),u:User=Depends(current_user)):
 ext=Path(file.filename or '').suffix.lower()
 if ext=='.csv': return {'sheets':[]}
 if ext not in ('.xlsx','.xls'): raise HTTPException(415,'Upload a CSV, XLSX, or XLS file')
 data=await file.read()
 if not data: raise HTTPException(400,'The uploaded file is empty')
 if len(data)>50*1024*1024: raise HTTPException(413,'Maximum file size is 50 MB')
 try: return {'sheets':pd.ExcelFile(io.BytesIO(data)).sheet_names}
 except Exception as e: raise HTTPException(422,'Workbook could not be read') from e
def dataset_view(d,df):
 return {'id':d.id,'filename':d.filename,'file_size':d.file_size,'row_count':d.rows,'column_count':d.columns,'sheet_name':d.sheet_name,'uploaded_at':d.uploaded_at.isoformat(),'columns':[{'name':str(c),'type':str(df[c].dtype),'null_count':int(df[c].isna().sum()),'unique_count':int(df[c].nunique(dropna=True))} for c in df.columns],'sample':json.loads(df.head(10).to_json(orient='records',date_format='iso'))}
@app.get('/api/datasets')
def datasets(u:User=Depends(current_user),db:Session=Depends(dbdep)):
 return [dataset_view(d,load_df(d)) for d in db.query(Dataset).filter_by(user_id=u.id).order_by(Dataset.uploaded_at.desc()).all()]
@app.get('/api/datasets/{dataset_id}')
def dataset(dataset_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db); return dataset_view(d,load_df(d))
@app.get('/api/datasets/{dataset_id}/preview')
def preview(dataset_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db); return {'columns':list(map(str,load_df(d).columns)),'rows':json.loads(load_df(d).head(100).to_json(orient='records',date_format='iso'))}
@app.delete('/api/datasets/{dataset_id}')
def delete_dataset(dataset_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db)
 for a in db.query(Analysis).filter_by(dataset_id=d.id).all(): db.query(Issue).filter_by(analysis_id=a.id).delete()
 db.query(CleaningOperation).filter_by(dataset_id=d.id).delete()
 db.query(Analysis).filter_by(dataset_id=d.id).delete(); Path(d.path).unlink(missing_ok=True)
 if d.cleaned_path: Path(d.cleaned_path).unlink(missing_ok=True)
 db.delete(d); db.commit(); return {'deleted':True}
def analyze(df,target_column=None,split_column=None,outlier_method='both',iqr_multiplier=1.5,zscore_threshold=3.0,custom_weights=None):
 issues=[]; profiles=[]; n=max(1,len(df))
 def issue(cat,col,desc,count,sev,examples,recommendation):
  issues.append({'category':cat,'column_name':str(col) if col is not None else None,'description':desc,'affected_count':int(count),'severity':sev,'examples':list(map(str,examples[:5])),'recommendation':recommendation,'status':'Open'})
 if len(df)==0: issue('Completeness',None,'Dataset contains no records.',0,'Critical',[],'Add valid data rows before using this dataset for training.')
 for c in df.columns:
  s=df[c]; null=int(s.isna().sum()); unique=int(s.nunique(dropna=True)); dtype=str(s.dtype)
  mode=s.mode(dropna=True); p={'column_name':str(c),'data_type':dtype,'null_count':null,'unique_count':unique,'mode':str(mode.iloc[0]) if not mode.empty else None,'statistics':{},'distribution':{str(k):int(v) for k,v in s.value_counts(dropna=True).head(10).items()},'type_distribution':{dtype:int(s.notna().sum())}}
  runtime_types=s.dropna().map(lambda value:type(value).__name__).value_counts()
  if len(runtime_types)>1:
   p['type_distribution']={str(k):int(v) for k,v in runtime_types.items()}
   issue('Consistency',c,'Column contains values represented by multiple Python data types.',int(runtime_types.iloc[1:].sum()),'Medium',s.dropna().head().tolist(),'Normalize the column to a consistent type after checking whether mixed values are valid.')
  if pd.api.types.is_numeric_dtype(s):
   p['statistics']={k:(None if v is None or pd.isna(v) else float(v)) for k,v in {'mean':s.mean(),'median':s.median(),'mode':mode.iloc[0] if not mode.empty else None,'std':s.std(),'min':s.min(),'max':s.max()}.items()}
   vals=s.dropna();
   if len(vals):
    q1,q3=vals.quantile([.25,.75]); iqr_mask=(s<q1-iqr_multiplier*(q3-q1))|(s>q3+iqr_multiplier*(q3-q1)); std=vals.std(); z_mask=((s-vals.mean()).abs()/std>zscore_threshold) if std and not pd.isna(std) else pd.Series(False,index=s.index)
    mask=iqr_mask if outlier_method=='iqr' else z_mask if outlier_method=='zscore' else iqr_mask|z_mask; cnt=int(mask.sum())
    if cnt: issue('Plausibility',c,f'{cnt} values flagged by {outlier_method} outlier detection (IQR Ã—{iqr_multiplier:g}, Z-score >{zscore_threshold:g}). Statistical flags are not confirmed errors.',cnt,'Medium',s[mask].head().tolist(),'Review flagged records against domain knowledge; do not remove automatically.')
  p['missing_percent']=round(null/n*100,2); profiles.append(p)
  if null: issue('Completeness',c,f'{null} missing values ({null/n*100:.1f}%).',null,'High' if null/n>.2 else 'Medium',s[s.isna()].index[:5].tolist(),'Consider median/mode imputation or removal based on column meaning and missingness.')
  if unique<=1 and len(s)>0: issue('Readiness',c,'Column is constant or contains no usable values.',len(s),'Medium',s.dropna().head().tolist(),'Remove the column or verify that it contains the intended feature.')
  if s.dtype=='object':
   text=s.dropna().astype(str); spaced=text[text.str.strip()!=text]
   if len(spaced): issue('Consistency',c,f'{len(spaced)} values have leading or trailing whitespace.',len(spaced),'Low',spaced.head().tolist(),'Trim whitespace and standardize text values.')
   normalized=text.str.strip().str.casefold(); variants={v:sorted(text[normalized==v].unique().tolist()) for v in normalized.unique()}
   inconsistent=[v for v,labels in variants.items() if len(labels)>1]
   if inconsistent:
    affected=text[text.str.strip().str.casefold().isin(inconsistent)]; issue('Consistency',c,f'{len(affected)} values use inconsistent capitalization or labels.',len(affected),'Low',affected.head().tolist(),'Review equivalent labels and standardize category spelling and casing.')
   if re.search(r'email|e_mail',str(c),re.I):
    bad=s.notna() & ~s.astype(str).str.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$'); cnt=int(bad.sum())
    if cnt: issue('Validity',c,f'{cnt} values do not match a basic email format.',cnt,'High',s[bad].head().tolist(),'Correct or remove malformed email addresses after review.')
   if re.search(r'phone|mobile|telephone',str(c),re.I):
    digits=s.fillna('').astype(str).str.replace(r'\D','',regex=True); bad=s.notna() & ~digits.str.len().between(7,15); cnt=int(bad.sum())
    if cnt: issue('Validity',c,f'{cnt} phone values do not contain 7â€“15 digits.',cnt,'Medium',s[bad].head().tolist(),'Check the country code and normalize phone number formats.')
   if re.search(r'date|time',str(c),re.I) and len(text):
    parsed=pd.to_datetime(s,errors='coerce',format='mixed'); bad=s.notna() & parsed.isna(); cnt=int(bad.sum())
    if cnt: issue('Validity',c,f'{cnt} values could not be parsed as dates.',cnt,'Medium',s[bad].head().tolist(),'Use a consistent date format and correct invalid dates.')
  if re.search(r'age',str(c),re.I):
   num=pd.to_numeric(s,errors='coerce'); bad=num<0; cnt=int(bad.sum())
   if cnt: issue('Validity',c,f'{cnt} ages are negative.',cnt,'High',s[bad].head().tolist(),'Verify source values; age should generally be non-negative.')
  if re.search(r'percent|percentage|pct',str(c),re.I):
   num=pd.to_numeric(s,errors='coerce'); bad=(num<0)|(num>100); cnt=int(bad.sum())
   if cnt: issue('Plausibility',c,f'{cnt} values fall outside the expected 0â€“100 percentage range.',cnt,'Medium',s[bad].head().tolist(),'Verify the percentage scale and review values outside 0â€“100.')
  if re.search(r'(^|[_\s])id($|[_\s])|identifier|_key$',str(c),re.I) and unique>0:
   duplicate=int(s.notna().sum()-unique)
   if duplicate: issue('Uniqueness',c,f'{duplicate} repeated values in an identifier-like column.',duplicate,'High',s[s.notna() & s.duplicated(keep=False)].head().tolist(),'Check ID generation and resolve collisions before training.')
 dup=int(df.duplicated().sum())
 if dup: issue('Uniqueness',None,f'{dup} fully duplicated rows.',dup,'High',df[df.duplicated()].head(3).index.tolist(),'Review duplicate records before model training.')
 weights={'Completeness':.25,'Uniqueness':.2,'Validity':.25,'Consistency':.15,'Plausibility':.15}
 if custom_weights:
  if set(custom_weights)-set(weights) or any(v<0 for v in custom_weights.values()): raise HTTPException(400,'Scoring weights must use supported dimensions and non-negative values')
  weights.update(custom_weights)
  total=sum(weights.values())
  if total<=0: raise HTTPException(400,'At least one scoring weight must be greater than zero')
  weights={k:v/total for k,v in weights.items()}
 penalties={k:0 for k in weights}
 for i in issues:
  dim=i['category']
  if dim in penalties: penalties[dim]=max(penalties[dim],min(.8,i['affected_count']/n*(.8 if i['severity']=='High' else .45)))
 scores={k:round(max(0,100*(1-min(.9,v))),1) for k,v in penalties.items()}; overall=round(sum(scores[k]*weights[k] for k in weights),1)
 category='Excellent' if overall>=90 else 'Good' if overall>=75 else 'Needs Improvement' if overall>=50 else 'Poor'
 numeric=df.select_dtypes(include=np.number); corr=[]
 if numeric.shape[1]>1:
  cm=numeric.corr().abs()
  for i,c in enumerate(cm.columns):
   for other in cm.columns[i+1:]:
    if cm.loc[c,other]>=.95: corr.append({'columns':[str(c),str(other)],'correlation':round(float(cm.loc[c,other]),3),'warning':'Review for redundancy; correlation alone does not establish leakage.'})
 constant=[p['column_name'] for p in profiles if p['unique_count']<=1]
 identifier=[p['column_name'] for p in profiles if len(df)>0 and p['unique_count']/len(df)>=.98 and p['unique_count']>10]
 readiness={'sample_count':len(df),'insufficient_sample':len(df)<100,'constant_columns':constant,'identifier_like_columns':identifier,'high_correlation_pairs':corr,'target_column':target_column,'target_missing_count':0,'class_distribution':{},'class_imbalance':None,'note':'Correlation and statistical checks are warnings, not proof of leakage or real-world inaccuracy.'}
 if len(df)<100: issues.append({'category':'Readiness','column_name':None,'description':f'Dataset has {len(df)} rows; this may be insufficient for a reliable model evaluation.','affected_count':len(df),'severity':'Informational','examples':[],'recommendation':'Confirm that the sample represents the intended population and use a validation strategy suitable for small datasets.','status':'Open'})
 if target_column and target_column in df.columns:
  target=df[target_column]; readiness['target_missing_count']=int(target.isna().sum()); dist=target.dropna().astype(str).value_counts()
  readiness['class_distribution']={str(k):int(v) for k,v in dist.items()}
  if len(dist)>1:
   ratio=float(dist.max()/max(1,dist.sum())); readiness['class_imbalance']=round(ratio,3)
   if ratio>=.8: issues.append({'category':'Readiness','column_name':str(target_column),'description':f'Target class distribution is imbalanced: largest class is {ratio:.1%} of non-missing labels.','affected_count':int(dist.max()),'severity':'Medium','examples':[f'{k}: {v}' for k,v in dist.head(5).items()],'recommendation':'Consider stratified splitting and class-weighting or resampling after evaluating your task.','status':'Open'})
  if readiness['target_missing_count']: issues.append({'category':'Readiness','column_name':str(target_column),'description':f"Target contains {readiness['target_missing_count']} missing labels.",'affected_count':readiness['target_missing_count'],'severity':'High','examples':[],'recommendation':'Resolve or exclude unlabeled target rows before supervised training.','status':'Open'})
 if split_column and split_column in df.columns:
  features=df.drop(columns=[split_column]).astype(str).agg('|'.join,axis=1); splits=df[split_column].astype(str); overlap=features.groupby(features).apply(lambda indexes: len(set(splits.loc[indexes.index]))>1)
  overlap_keys=set(overlap[overlap].index.tolist()) if len(overlap) else set(); mask=features.isin(overlap_keys)
  readiness['split_column']=split_column; readiness['cross_split_duplicate_rows']=int(mask.sum())
  if mask.any(): issues.append({'category':'Readiness','column_name':split_column,'description':f'{int(mask.sum())} identical feature rows appear in more than one split. Review for possible train/test overlap.','affected_count':int(mask.sum()),'severity':'High','examples':df.index[mask].astype(str).tolist()[:5],'recommendation':'Remove overlap or confirm that repeated records across splits are intentional.','status':'Open'})
 return {'overall_score':overall,'quality_category':category,'scores':scores,'weights':weights,'issues':issues,'profiles':profiles,'readiness':readiness,'recommendations':[i['recommendation'] for i in issues]}
@app.post('/api/analyses/{dataset_id}/run')
def run_analysis(dataset_id:int,body:AnalysisRequest|None=None,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db); result=analyze(load_df(d),body.target_column if body else None,body.split_column if body else None,body.outlier_method if body else 'both',body.iqr_multiplier if body else 1.5,body.zscore_threshold if body else 3.0,body.weights if body else None); a=Analysis(dataset_id=d.id,score=result['overall_score'],category=result['quality_category'],result=json.dumps(result)); db.add(a); db.commit(); db.refresh(a)
 for i in result['issues']: db.add(Issue(analysis_id=a.id,category=i['category'],column_name=i['column_name'],description=i['description'],severity=i['severity'],affected_count=i['affected_count'],examples=json.dumps(i['examples']),recommendation=i['recommendation']))
 db.commit(); return {'id':a.id,'dataset_id':d.id,'created_at':a.created_at.isoformat(),**result}
@app.get('/api/analyses/{analysis_id}')
def get_analysis(analysis_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 a=db.get(Analysis,analysis_id)
 if not a or not db.query(Dataset).filter_by(id=a.dataset_id,user_id=u.id).first(): raise HTTPException(404,'Analysis not found')
 return {'id':a.id,'dataset_id':a.dataset_id,'created_at':a.created_at.isoformat(),**json.loads(a.result)}
@app.get('/api/analyses')
def analysis_history(skip:int=0,limit:int=50,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 limit=max(1,min(limit,100)); rows=db.query(Analysis).join(Dataset).filter(Dataset.user_id==u.id).order_by(Analysis.created_at.desc()).offset(max(skip,0)).limit(limit).all()
 return [{'id':a.id,'dataset_id':a.dataset_id,'filename':db.get(Dataset,a.dataset_id).filename,'score':a.score,'category':a.category,'created_at':a.created_at.isoformat()} for a in rows]
@app.get('/api/analyses/{analysis_id}/issues')
def issues(analysis_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 a=db.get(Analysis,analysis_id)
 if not a or not db.query(Dataset).filter_by(id=a.dataset_id,user_id=u.id).first(): raise HTTPException(404,'Analysis not found')
 return [{'id':x.id,'category':x.category,'column_name':x.column_name,'description':x.description,'severity':x.severity,'affected_count':x.affected_count,'examples':json.loads(x.examples),'recommendation':x.recommendation,'status':x.status} for x in db.query(Issue).filter_by(analysis_id=analysis_id).all()]
@app.get('/api/issues')
def all_issues(skip:int=0,limit:int=100,severity:str|None=None,category:str|None=None,status:str|None=None,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 q=db.query(Issue).join(Analysis).join(Dataset).filter(Dataset.user_id==u.id)
 if severity: q=q.filter(Issue.severity==severity)
 if category: q=q.filter(Issue.category==category)
 if status: q=q.filter(Issue.status==status)
 rows=q.order_by(Issue.id.desc()).offset(max(skip,0)).limit(max(1,min(limit,200))).all()
 return [{'id':x.id,'analysis_id':x.analysis_id,'dataset_id':db.get(Analysis,x.analysis_id).dataset_id,'category':x.category,'column_name':x.column_name,'description':x.description,'severity':x.severity,'affected_count':x.affected_count,'examples':json.loads(x.examples),'recommendation':x.recommendation,'status':x.status} for x in rows]
@app.patch('/api/issues/{issue_id}/status')
def issue_status(issue_id:int,body:StatusUpdate,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 if body.status not in ['Open','Reviewed','Resolved','Ignored']: raise HTTPException(400,'Invalid issue status')
 x=db.get(Issue,issue_id)
 if not x or not db.query(Dataset).join(Analysis).filter(Analysis.id==x.analysis_id,Dataset.user_id==u.id).first(): raise HTTPException(404,'Issue not found')
 x.status=body.status; db.commit(); return {'id':x.id,'status':x.status}
def safe_csv_value(value):
 if isinstance(value,str) and re.match(r'^\s*[=+@\-]',value): return "'"+value
 return value
def csv_stream(frame,filename):
 safe=frame.copy()
 for col in safe.columns:
  if pd.api.types.is_object_dtype(safe[col]): safe[col]=safe[col].map(safe_csv_value)
 out=io.StringIO(); safe.to_csv(out,index=False)
 return StreamingResponse(iter([out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename={filename}'})
@app.get('/api/analyses/{analysis_id}/report/{kind}')
def report(analysis_id:int,kind:str,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 a=db.get(Analysis,analysis_id)
 if not a or not db.query(Dataset).filter_by(id=a.dataset_id,user_id=u.id).first(): raise HTTPException(404,'Analysis not found')
 dataset_record=db.get(Dataset,a.dataset_id); analysis_payload=json.loads(a.result)
 status_map={(x.category,x.column_name,x.description):x.status for x in db.query(Issue).filter_by(analysis_id=a.id).all()}
 for item in analysis_payload.get('issues',[]): item['status']=status_map.get((item['category'],item.get('column_name'),item['description']),'Open')
 payload={'dataset':dataset_record.filename,'analysis':analysis_payload,'cleaning_history':[{'operations':json.loads(x.operation_details),'before':json.loads(x.before_stats),'after':json.loads(x.after_stats),'created_at':x.created_at.isoformat()} for x in db.query(CleaningOperation).filter_by(dataset_id=a.dataset_id).order_by(CleaningOperation.created_at.asc()).all()],'generated_at':utcnow().isoformat()+'Z'}
 if kind=='json': return StreamingResponse(io.BytesIO(json.dumps(payload,indent=2).encode()),media_type='application/json',headers={'Content-Disposition':f'attachment; filename=analysis-{analysis_id}.json'})
 if kind=='csv':
  result=payload['analysis']; records=[{'record_type':'summary','dataset':payload['dataset'],'overall_score':a.score,'quality_category':a.category,'dimension_scores':json.dumps(result['scores']),'generated_at':payload['generated_at']}]
  records.extend({'record_type':'issue',**item} for item in result['issues']); records.extend({'record_type':'column_profile',**item} for item in result['profiles']); records.extend({'record_type':'readiness','finding':json.dumps(result['readiness'])} for _ in [0])
  return csv_stream(pd.DataFrame(records),'analysis-'+str(analysis_id)+'.csv')
 if kind=='pdf':
  try:
   from reportlab.pdfgen import canvas
   from reportlab.lib.pagesizes import letter
   buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=letter); y=750
   result=payload['analysis']
   lines=[f"DataGuard AI Report: {payload['dataset']}",f"Score: {a.score}/100 ({a.category})",f"Generated: {payload['generated_at']}","Dimension scores: "+json.dumps(result['scores']),"",'ISSUES']
   lines += [f"{i['severity']} | {i['category']} | {i.get('column_name') or 'Dataset'}: {i['description']} | Recommendation: {i['recommendation']}" for i in result['issues']]
   lines += ["",'COLUMN PROFILES']
   lines += [f"{p['column_name']} ({p['data_type']}): nulls={p['null_count']}, unique={p['unique_count']}, statistics={json.dumps(p['statistics'])}" for p in result['profiles']]
   lines += ["",'ML READINESS',json.dumps(result['readiness']),"",'CLEANING HISTORY',json.dumps(payload['cleaning_history'])]
   for line in lines:
    if y<60: c.showPage(); y=750
    c.drawString(40,y,line[:110].encode('cp1252','replace').decode('cp1252')); y-=18
   c.save(); buf.seek(0); return StreamingResponse(buf,media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename=analysis-{analysis_id}.pdf'})
  except ImportError: raise HTTPException(501,'PDF support is unavailable; install reportlab')
 raise HTTPException(404,'Report format must be pdf, csv, or json')
def cleaned_frame(source,body):
 df=source.copy(); operations=[]
 if body.missing_strategy not in ['none','drop_rows','mean','median','mode','constant']: raise HTTPException(400,'Invalid missing value strategy')
 if body.remove_empty_columns: df=df.dropna(axis=1,how='all'); operations.append('remove_empty_columns')
 if body.trim_whitespace:
  for c in df.select_dtypes(include=['object','str']): df[c]=df[c].map(lambda x:x.strip() if isinstance(x,str) else x)
  operations.append('trim_whitespace')
 if body.text_case!='none':
  for c in df.select_dtypes(include=['object','str']):
   df[c]=df[c].map(lambda x:x.lower() if isinstance(x,str) and body.text_case=='lower' else x.upper() if isinstance(x,str) and body.text_case=='upper' else x.title() if isinstance(x,str) else x)
  operations.append('text_case:'+body.text_case)
 if body.remove_duplicates: df=df.drop_duplicates(); operations.append('remove_duplicates')
 for column,bounds in body.range_rules.items():
  if column not in df.columns: raise HTTPException(400,f'Range rule column not found: {column}')
  if 'min' in bounds and 'max' in bounds and float(bounds['min'])>float(bounds['max']): raise HTTPException(400,f'Invalid range for column: {column}')
  numeric=pd.to_numeric(df[column],errors='coerce'); outside=pd.Series(False,index=df.index)
  if 'min' in bounds: outside|=numeric<float(bounds['min'])
  if 'max' in bounds: outside|=numeric>float(bounds['max'])
  if outside.any(): df.loc[outside,column]=np.nan
  operations.append(f"range:{column}:{json.dumps(bounds,sort_keys=True)}")
 if body.missing_strategy=='drop_rows': df=df.dropna(); operations.append('missing:drop_rows')
 elif body.missing_strategy in ['mean','median','mode','constant']:
  for c in df.columns:
   if df[c].isna().any():
    value=body.constant_value if body.missing_strategy=='constant' else (df[c].mean() if body.missing_strategy=='mean' and pd.api.types.is_numeric_dtype(df[c]) else df[c].median() if body.missing_strategy=='median' and pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0] if not df[c].mode().empty else '')
    df[c]=df[c].fillna(value)
  operations.append('missing:'+body.missing_strategy)
 return df,operations
def clean_summary(before,after,operations):
 removed=before['rows']-after['rows']
 return {'before':before,'after':after,'removed_rows':removed,'removed_columns':before['columns']-after['columns'],'warning':'This operation removes more than 20% of rows.' if before['rows'] and removed/before['rows']>.2 else None,'operations':operations}
@app.post('/api/datasets/{dataset_id}/clean/preview')
def preview_clean(dataset_id:int,body:CleanRequest,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db); original=load_df(d); proposed,operations=cleaned_frame(original,body)
 before={'rows':len(original),'columns':len(original.columns)}; after={'rows':len(proposed),'columns':len(proposed.columns)}
 return {**clean_summary(before,after,operations),'sample_before':json.loads(original.head(5).to_json(orient='records',date_format='iso')),'sample_after':json.loads(proposed.head(5).to_json(orient='records',date_format='iso'))}
@app.post('/api/datasets/{dataset_id}/clean')
def clean(dataset_id:int,body:CleanRequest,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db); original=load_df(d); df,operations=cleaned_frame(original,body); before={'rows':len(original),'columns':len(original.columns)}
 ext='.csv' if d.file_type=='csv' else '.xlsx'; path=UPLOADS/f'{secrets.token_hex(16)}-clean{ext}'
 if ext=='.csv':
  safe=df.copy()
  for col in safe.columns:
   if pd.api.types.is_object_dtype(safe[col]): safe[col]=safe[col].map(safe_csv_value)
  safe.to_csv(path,index=False)
 else:
  safe=df.copy()
  for col in safe.columns:
   if pd.api.types.is_object_dtype(safe[col]): safe[col]=safe[col].map(safe_csv_value)
  safe.to_excel(path,index=False)
 if d.cleaned_path: Path(d.cleaned_path).unlink(missing_ok=True)
 d.cleaned_path=str(path); after={'rows':len(df),'columns':len(df.columns)}
 db.add(CleaningOperation(dataset_id=d.id,operation_details=json.dumps(operations),before_stats=json.dumps(before),after_stats=json.dumps(after))); db.commit()
 return {**clean_summary(before,after,operations),'download_url':f'/api/datasets/{d.id}/download-cleaned'}
@app.get('/api/datasets/{dataset_id}/download-cleaned')
def download_clean(dataset_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db)
 if not d.cleaned_path or not Path(d.cleaned_path).exists(): raise HTTPException(404,'No cleaned dataset is available')
 return StreamingResponse(open(d.cleaned_path,'rb'),media_type='application/octet-stream',headers={'Content-Disposition':f'attachment; filename=cleaned-{d.filename}'})
@app.get('/api/datasets/{dataset_id}/cleaning-history')
def cleaning_history(dataset_id:int,u:User=Depends(current_user),db:Session=Depends(dbdep)):
 d=owned_dataset(dataset_id,u,db); return [{'id':x.id,'operations':json.loads(x.operation_details),'before':json.loads(x.before_stats),'after':json.loads(x.after_stats),'created_at':x.created_at.isoformat()} for x in db.query(CleaningOperation).filter_by(dataset_id=d.id).order_by(CleaningOperation.created_at.desc()).all()]
@app.get('/api/dashboard/summary')
def dashboard(u:User=Depends(current_user),db:Session=Depends(dbdep)):
 ds=db.query(Dataset).filter_by(user_id=u.id).all(); analyses=[]
 for d in ds: analyses.extend(db.query(Analysis).filter_by(dataset_id=d.id).order_by(Analysis.created_at.desc()).all())
 ordered=sorted(analyses,key=lambda item:item.created_at,reverse=True); chronological=list(reversed(ordered))
 issue_rows=db.query(Issue).join(Analysis).join(Dataset).filter(Dataset.user_id==u.id).all(); issues_count=len(issue_rows); critical=sum(i.severity=='Critical' for i in issue_rows)
 issue_distribution={}
 for item in issue_rows: issue_distribution[item.category]=issue_distribution.get(item.category,0)+1
 quality_categories={}
 for item in analyses: quality_categories[item.category]=quality_categories.get(item.category,0)+1
 missing_values={}
 for d in ds:
  latest=db.query(Analysis).filter_by(dataset_id=d.id).order_by(Analysis.created_at.desc()).first()
  if latest:
   for p in json.loads(latest.result).get('profiles',[]): missing_values[p['column_name']]=missing_values.get(p['column_name'],0)+p['null_count']
 recent=[{'id':a.id,'dataset_id':a.dataset_id,'filename':db.get(Dataset,a.dataset_id).filename,'score':a.score,'category':a.category,'created_at':a.created_at.isoformat()} for a in ordered[:8]]
 return {'total_datasets':len(ds),'analyses_count':len(analyses),'average_score':round(sum(a.score for a in analyses)/len(analyses),1) if analyses else None,'total_issues':issues_count,'critical_issues':critical,'issue_distribution':issue_distribution,'quality_categories':quality_categories,'missing_values':[{'column':k,'missing':v} for k,v in sorted(missing_values.items(),key=lambda x:x[1],reverse=True)[:10]],'recent':recent,'quality_trends':[{'date':a.created_at.date().isoformat(),'score':a.score} for a in chronological[-20:]]}
@app.get('/api/dashboard/quality-trends')
def trends(u:User=Depends(current_user),db:Session=Depends(dbdep)): return dashboard(u,db)['quality_trends']
