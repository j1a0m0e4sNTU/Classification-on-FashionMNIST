python3 main.py train base_02 -lr 0.001 -batch_size 32 -epoch_num 20 -save weights/base_02.pkl -log logs/base_02.txt -check_batch_num 200

python3 main.py train res_01  -lr 0.001 -batch_size 32 -epoch_num 20 -save weights/res_01.pkl  -log logs/res_01.txt  -check_batch_num 200
