# A python script reproducing the results in the paper

### Clone the repository
```
git clone https://github.com/seznam/vertical-search-blending-dataset.git
cd vertical-search-blending-dataset
```

### Get the dataset
(about 5 GB uncompressed)
```
mkdir dataset
cd dataset

for i in 0 1 2 ; do
    wget https://github.com/seznam/vertical-search-blending-dataset/releases/latest/download/part${i}.tar.gz
    tar -zxvf part${i}.tar.gz
    rm part${i}.tar.gz
done

cd ..
```

### Install Vowpal Wabbit
https://github.com/VowpalWabbit/vowpal_wabbit/

We used version 8.6.1.

### Create a new conda environment (optional)
```
conda create --name vsbd pip
conda activate vsbd
```

### Install python dependencies
```
pip install -r experiments/python/requirements.txt
```

### Run the experiment
```
mkdir experiments/results
python experiments/python/vowpal_experiment.py --vowpal_path /path/to/the/vowpal/binary --dataset_dir dataset --results_dir experiments/results --num_test_positions 4 --cb_types dm ips dr
```
and inspect the directory created in `experiments/results`. Optionally with flexp-browser:
```
cd experiments/results
flexp-browser -p 9999
```
and navigate to http://localhost:9999.
