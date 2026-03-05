运行顺序
python bilstm_fakenews.py        # 训练BiLSTM，生成results/bilstm_results.json
python bert_bilstm_fakenews.py   # 训练BERT+BiLSTM，生成results/bert_bilstm_results.json
python compare_models.py         # 读取两个JSON，生成对比图
