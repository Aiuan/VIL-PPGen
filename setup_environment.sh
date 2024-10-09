conda create -n VIL-PPGen python=3.9

conda activate VIL-PPGen

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

conda install pandas matplotlib tqdm scipy scikit-learn rich openpyxl

pip install opencv-python tensorboard pyyaml

pip install open3d==0.16.0

pip install spconv-cu117

python setup.py develop