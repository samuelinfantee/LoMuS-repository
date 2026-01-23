# LoMuS-repository

# Running Tsuboyama dataset:  
step 1:   
```
conda activate /data/sinfante/envs/tape  
pip install --no-input "datasets==2.20.0" "pyarrow>=10" fsspec  
```

step 2:  
```
cd ~/protstab/external/EsmTherm  
mv datasets esmt_data 2>/dev/null || true  
ln -s esmt_data datasets 2>/dev/null || true   
[ -f requirements.txt ] && pip install --no-input -r requirements.txt  
pip install -e .
```

step 3:  
```
cd ~/protstab/external/EsmTherm  
python prebuild_dataset.py  
```
```
python build_dataset.py \  
  --dataset_dir esmt_data/dataset \  
  --csv        esmt_data/analysis/filtered_data.csv \  
  --split_csv  esmt_data/wildtype_split.csv  
```
```
ls -lh esmt_data/dataset
```  
(expect: args.json, dataset.csv, dataset_dict.json, and dirs: train  val  test)  

step 4:
```
python ~/protstab/scripts/esmtherm_to_lomus.py \    
  --split_dir  ~/protstab/external/EsmTherm/esmt_data/dataset \  
  --out_root   ~/protstab/data/dms_one/tsub_mega  

head -n 3 ~/protstab/data/dms_one/tsub_mega/train.csv  
```

step 5:
Run the features_CSV.py code to generate the per-sequence features:
```
cd ~/protstab  
python features.py --root data/dms_one --protein tsub_mega  
```

step 6:
Submit the SLURM job that runs both training and testing:
```
sbatch run_tsubo.slurm
```

# Running TAPE stability:
Step 1:  
Dowload and extract the stability dataset:  
```
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz  
tar -xvzf stability.tar.gz -C ./data/  
```

Directory contents after extraction:  
./data/stability/  
stability_train.lmdb/  
stability_valid.lmdb/  
stability_test.lmdb/  

Step 2:
Run the features_FASTA.py code to generate the per-sequence features:
```
python features_FASTA.py
```
step 3:
Submit the SLURM job that runs both training and testing:
```
sbatch run_tape.slurm
```

# Running DMS proteins:  
Step 1:  
```
cd ~/protstab  
mkdir -p data/proteingym && cd data/proteingym  
curl -L -o DMS_ProteinGym_substitutions.zip https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip  
```

step 2:
```
python tools/prepare_unirep_dms.py --raw data/proteingym --out data/dms --seed 42 --min_seqs 200  
```

step 3 (train on a specific protein):  
```
cd ~/protstab  
mkdir -p data/dms_one  
rsync -a data/dms/[PROTEIN_NAME]/ data/dms_one/[PROTEIN NAME]/
```
(proteins used in LoMuS paper: YAP1_HUMAN_Araya_2012, VILI_CHICK_Tsuboyama_2023_1YU5, PIN1_HUMAN_Tsuboyama_2023_1I6C)  
```
python features_CSV.py  
sed -i 's|^DMS_ROOT\s*=.*|DMS_ROOT     = "./data/dms_one"|' features_CSV.py
```
to remove a protein (after running features_csv.py): 
```
rm -rf data/dms_one/ [PROTEIN_NAME]  
```
step 4:
Submit the SLURM job that runs both training and testing:
```
sbatch run_dms.slurm
```
Before submitting, make sure the protein ID in the SLURM script matches the protein you used in features_CSV:
```
# --- choose your protein ---
PROTEIN="PIN1_HUMAN_Tsuboyama_2023_1I6C" #Or any other
```



