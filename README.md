# MuRaStab-repository

# Running DMS proteins:
Step 1:  
cd ~/protstab  
mkdir -p data/proteingym && cd data/proteingym  
curl -L -o DMS_ProteinGym_substitutions.zip https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip  

step 2:  
python prepare_unirep_dms.py  
python tools/prepare_unirep_dms.py --raw data/proteingym --out data/dms --seed 42 --min_seqs 200  

step 3 (train on a specific protein):  
cd ~/protstab  
mkdir -p data/dms_one  
rsync -a data/dms/[PROTEIN_NAME]/ data/dms_one/[PROTEIN NAME]/  
(proteins used in MuRaStab paper: YAP1_HUMAN_Araya_2012, VILI_CHICK_Tsuboyama_2023_1YU5, PIN1_HUMAN_Tsuboyama_2023_1I6C)  

python features_csv.py  
sed -i 's|^DMS_ROOT\s*=.*|DMS_ROOT     = "./data/dms_one"|' features.py  
