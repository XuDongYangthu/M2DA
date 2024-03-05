## Contents
1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Data Generation](#data-generation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Acknowledgements](#acknowledgements)

## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
git clone git@github.com:XuDongYangthu/AD_competition.git
cd AD_competition
conda create -n AD_competition python=3.7
conda activate AD_competition
pip3 install -r requirements.txt
cd my_model
python setup.py develop
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
./setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```

**Note:** we choose the setuptools==41 to install because this version has the feature `easy_install`. After installing the carla.egg you can install the lastest setuptools to avoid *No module named distutils_hack*.

**add** the checkpoint to **AD_competition/my_model/timm/weights**
- [weights_best.pth](https://cloud.tsinghua.edu.cn/f/9f42b1b6795644ad8c8a/?dl=1)

## Dataset
The data is generated with ```leaderboard/team_code/autopilot.py``` in 8 CARLA towns using the routes and scenarios files provided at ```leaderboard/data``` on CARLA 0.9.10.1

The collected dataset is structured as follows:
```
- TownX_{tiny,short,long}: corresponding to different towns and routes files
    - routes_X: contains data for an individual route
        - rgb_{front, left, right, rear}: multi-view camera images at 400x300 resolution
        - seg_{front, left, right}: corresponding segmentation images
        - depth_{front, left, right}: corresponding depth images
        - lidar: 3d point cloud in .npy format
        - birdview: topdown segmentation images required for training LBC
        - 2d_bbs_{front, left, right, rear}: 2d bounding boxes for different agents in the corresponding camera view
        - 3d_bbs: 3d bounding boxes for different agents
        - affordances: different types of affordances
        - measurements: contains ego-agent's position, velocity and other metadata
        - other_actors: contains the positions, velocities and other metadatas of surrounding vehicles and the traffic lights
```

## Data Generation
### Data Generation with a CARLA Server
With a single CARLA server, rollout the autopilot to start data generation.
```Shell
carla/CarlaUE4.sh --world-port=2000 -opengl
./leaderboard/scripts/run_evaluation.sh
```
The expert agent used for data generation is defined in ```leaderboard/team_code/autopilot.py```. Different variables which need to be set are specified in ```leaderboard/scripts/run_evaluation.sh```. 


## Training

```bash
cd M2DA
bash scripts/train.sh
```
where the train.sh is shown follows:
```bash
GPU_NUM=8
DATASET_ROOT=$Your_dataset   #such as : 'dataset/'


 
./distributed_train_raw1.sh $GPU_NUM $DATASET_ROOT  --dataset carla     
    --train-towns 1 2 3 4 6 7 10  --val-towns 5 
    --train-weathers 0 1 2 3 4 5 6 7 8 9 --val-weathers 10 11 12 13 
    --model M2DA --sched cosine --epochs 25  --warmup-epochs 5 
    --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher 
    --eval-metric l1_error --opt adamw --opt-eps 1e-8 
    --weight-decay 0.05  --scale 0.9 1.1 --saver-decreasing 
    --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment M2DA \
    --pretrained
```

The `DATASET_ROOT` needs a file dataset_index.txt to list the traing/evaluation data. 

Readers can use write_index.py within the path of your dataset, we suggest that the init_dir use absolute path:

```python
import os
init_dir = Your_dataset #'dataset/'

for dirs in os.listdir(init_dir):
    if os.path.basename(dirs) == 'data':
        data_dir = os.path.join(init_dir, dirs)
        for sub_dirs in os.listdir(data_dir):
            list_dir = os.listdir(os.path.join(data_dir,sub_dirs))
            rgb_dir = list_dir[0]
            length = len(os.listdir(list_dir, rgb_dir))
            name = (sub_dirs) + '/ ' + str(length) + '\n'
            with open('dataset_index.txt', mode='a') as f:
                f.write(name)

```

## Evaluation
Spin up a CARLA server (described above) and run the required agent. The adequate routes and scenarios files are provided in ```leaderboard/data``` and the required variables need to be set in ```leaderboard/scripts/run_evaluation.sh```.
Update ```leaderboard/scripts/run_evaluation.sh``` to include the following code for evaluating the model on Town05 Long Benchmark.
```shell
export CARLA_ROOT=/path/to/carla/root
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml
export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export TEAM_AGENT=leaderboard/team_code/my_agent.py
export TEAM_CONFIG=leaderboard/team_code/my_config.py
export CHECKPOINT_ENDPOINT=results/result.json
```

CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh

Yu can use `ROUTES=leaderboard/data/longest6.xml` and `SCENARIOS=leaderboard/data/longest6.json` to run Longest6 Benchmark.


the checkpoint as follows:
- [model is q_all_feature_noshijue](https://cloud.tsinghua.edu.cn/f/762cc0871c7c4d26a303/?dl=1)




## Acknowledgements
This implementation is based on code from several repositories.
- [Interfuser](https://github.com/opendilab/InterFuser.git)
- [Transfuser](https://github.com/autonomousvision/transfuser)
- [2020_CARLA_challenge](https://github.com/bradyz/2020_CARLA_challenge)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)

