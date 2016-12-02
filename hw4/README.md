# 相依軟件
- gensim (https://radimrehurek.com/gensim/)
- scikit-learn (http://scikit-learn.org/stable/)
- pandas (http://pandas.pydata.org)
- Google's word2vec (https://github.com/danielfrg/word2vec)

# 執行程序腳本
```
bash cluster.sh <path_to_data_folder> <output_csv_filename>
```

# 運行平台與時間
- 操作系統: Ubuntu 16.04 LTS (64-bit)
- CPU: Intel® Core™ i7-6700K (4.00GHz × 8)
- 記憶體: 15.6 GiB
- 固態硬碟
- 共約 3 分 07 秒 (前處理和訓練 K-means 很快，時間多花在預測 test cases，約 2 分 40 秒)
- 同樣方法多次上傳 Kaggle 分數約在 0.89 ~ 0.91 間變化
