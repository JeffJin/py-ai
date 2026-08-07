
On the remote, restart Jupyter with:
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

On Mac
ssh -N -L 8888:localhost:8888 jeffjin@10.0.0.133

Click the + button (top of the Servers list)
A new entry will appear — change its type to "Configured Server" (not IDE-Managed)
Set the URL to: