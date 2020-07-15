@echo on

python SeaNMF_train.py --n_topics 15 --alpha 0.33 --beta 1.0
python SeaNMF_train.py --n_topics 15 --alpha 0.66 --beta 1.0
python SeaNMF_train.py --n_topics 15 --alpha 1.0 --beta 1.0
python SeaNMF_train.py --n_topics 15 --alpha 0.33 --beta 2.0
python SeaNMF_train.py --n_topics 15 --alpha 0.66 --beta 2.0
python SeaNMF_train.py --n_topics 15 --alpha 1.0 --beta 2.0


timeout /t -1
