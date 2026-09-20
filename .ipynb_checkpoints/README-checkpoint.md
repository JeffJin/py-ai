
On the remote, restart Jupyter with:
jupyter notebook stop 8888
pkill -f jupyter

conda activate xcs-torch
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

On Mac
ssh -N -L 8888:localhost:8888 jeffjin@10.0.0.133

Click the + button (top of the Servers list)
A new entry will appear — change its type to "Configured Server" (not IDE-Managed)
Set the URL to:

MUTUGEN SETUP

ssh-add --apple-use-keychain ~/.ssh/id_ed25519

mutagen daemon register 2>&1
mutagen daemon register

mutagen daemon stop 2>&1; sleep 2; ps aux | grep -i mutagen | grep -v grep; echo "---register---"; mutagen daemon register 2>&1

mutagen sync list — check status
mutagen sync pause py-ai / resume py-ai — pause/resume
mutagen sync terminate py-ai — stop sync completely