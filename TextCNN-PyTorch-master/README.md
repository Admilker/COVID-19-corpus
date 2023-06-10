
中文文本分类，基于pytorch。

- 神经网络模型：TextCNN

参考：
- [ERNIE - 详解](https://baijiahao.baidu.com/s?id=1648169054540877476)
- [DPCNN 模型详解](https://zhuanlan.zhihu.com/p/372904980)
- [从经典文本分类模型TextCNN到深度模型DPCNN](https://zhuanlan.zhihu.com/p/35457093)

## 环境
python 3.7  
pytorch 1.1   —— 使用GPU版本 添加conda清华源下载
tqdm  
openpyxl
sklearn  
tensorboardX

# 训练并测试：
# TextCNN
python run.py --model TextCNN


主入口文件：run.py
测试：predict.py
计算KL：kl.py
