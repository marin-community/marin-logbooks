import os, subprocess, sys
for name in ('CW_KEY_ID','CW_KEY_SECRET'):
 result=subprocess.run(['gcloud','secrets','versions','access','latest','--project','hai-gcp-models','--secret','BELAY_'+name],capture_output=True,text=True,check=True)
 os.environ[name]=result.stdout.strip()
os.execvp(sys.argv[1],sys.argv[1:])
