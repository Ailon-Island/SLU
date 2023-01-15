### 创建环境

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1
    pip install allennlp, pytorch-crf
    
    并安装ELMo for manylangs
    git clone https://github.com/ericxsun/ELMoForManyLangs.git
    cd ELMoForManyLangs
    python setup.py install

    如果您想运行SLU-GNN的代码，请安装pyg库。

### 运行测试代码
    
在根目录下运行以下代码可以实现对ELMO+SLSTM+CRF的调用：

    python scripts/slu_baseline_elmo.py

### 代码说明
除了原脚本外，我们添加了以下文件内容：

+ `model/attention.py`:实现了注意力机制模型
+ `model/graphlstm.py`:实现了GraphLSTM的代码
+ `model/slstm.py`:实现了Sequence-State LSTM的代码
+ `model/slugnn.py`:实现了SLU-GNN的代码
+ `model/stack_propagation.py`:实现了Stack-Propagation（LSTM-decoder）的代码
+ `model/bilstm_crf`:双向LSTM+CRF的最初实现，CRF算法耗时较长
+ `model/bilstmcrf.py`:使用torchcrf库优化后的BiLSTM+CRF实现
+ `model/bilstm_crf.py`:添加了ELMo模型的BiLSTM+CRF算法
+ `scripts/correct.py`:纠错数据集的生成代码
+ `scripts/train_crf_elmo_correct.py`:使用纠错后数据集的ELMo+BiLSTM+CRF训练
+ `scripts/train_crf_elmo.py`:ELMo+BiLSTM+CRF训练
+ `scripts/train_crf.py`:BiLSTM+CRF训练

+ `scripts/slu_basline_elmo.py`:实现了以上模型的训练接口，请注意若想调用请自行修正代码，我们没有提供接口

