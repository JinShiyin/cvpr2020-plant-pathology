conda create -n fgcv python=3.8
conda init 
conda activate fgcv

pip install -r requirements.txt  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
conda install -c conda-forge nvidia-apex