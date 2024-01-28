```
conda create -n outpaint python==3.10 -y
conda activate outpaint 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```