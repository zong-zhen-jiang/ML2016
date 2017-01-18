# 運行環境
- 操作系統: Ubuntu 16.04 (64-bit)
- 記憶體: 25 GB 以上 (或包含 swap 25 GB 以上)

# 相依套件
- pandas (http://pandas.pydata.org)
- PyPy (http://pypy.org)
- 支援 C++11 的 g++ 編譯器
- POSIX Pthread

# 執行程式
請把所有 Outbrain 檔案都放在 `data/` 資料夾下，並切換到 `src/` 下：
```
cd src
```
然後依下面步驟執行，包含前處理、訓練、驗證等，共 5 步驟。步驟 1 ~ 3 雖耗資源，但通常只執行一次，我們多是在步驟 4 嘗試修改參數或刪除 features。
## 1. 產生 Leak 檔
請執行：
```
pypy leak.py
```
會生成 `data/_leak.csv` 檔案。約需 10 GB 記憶體，跑 20 ~ 30 分鐘。
## 2. 切 Validation
若要依照 testing set 時間分佈來切，請執行：
```
pypy splitvalid.py
```
若只是要隨機切，請執行：
```
pypy splitvalidrand.py
```
約需 3 GB 記憶體。
## 3. 抓 Features
請執行：
```
pypy features.py
```
約需 25 GB 記憶體，跑 20 ~ 30 分鐘。
## 4. 訓練 FTRL 模型
除了我們自己實作的 C++ 版本，為做比較我們也提供原始 Python 版本。我們的版本需先編譯再執行，指令如下：
```
g++ main.cpp -std=c++11 -lpthread -O3
./a.out
```
每個 epoch 只需約 50 秒。這邊會產出 `data/_va_out.csv` 和 `data/_te_out.csv`，分別是 validation 和 testing 的結果，後者可上傳 Kaggle。注意實際上我們若要上傳 Kaggle，則會改用全部 training 資料再重新訓練一次，也就是把 `main.cpp` 第 44 行註解掉改為第 45 行。<br>
<br>
若要執行原 Python 版本，則指令如下：
```
pypy ftrl.py
```
每個 epoch 約要 600 秒，而且要超過 20 GB 記憶體。
## 5. 計算 Validation 分數
請執行：
```
python validate.py
```
會從 4. 產生的 validation 結果，計算 MAP@12 分數。
