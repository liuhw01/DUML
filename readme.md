DUML
===

This is the source code for paper

*Hanwei Liu, Huiling Cai, Qingcheng Lin, Xuefeng Li, Hui Xiao. Learning from More: Combating Uncertainty Cross-multidomain for Facial Expression Recognition, MM '23: Proceedings of the 31th ACM International Conference on Multimedia *


## Inference

* Step 1: In order to infer your results, you should store the dataset according to the path described in the txt file in `./data/label_multi`

The prepared file format is as follows:

```
./data
	./CK
	./AffectNet
	./FER2013
	./JAFFE
	./RAF-DB
	./Oulu_CASIA

```


* Step 2:  run `inference.py`.


* Training logs have been saved in `./recorder_pre`.
