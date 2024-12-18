DUML
===

This is the source code for paper

*Hanwei Liu, Huiling Cai, Qingcheng Lin, Xuefeng Li, Hui Xiao. [Learning from More: Combating Uncertainty Cross-multidomain for Facial Expression Recognition]{https://dl.acm.org/doi/10.1145/3581783.3611702}, MM '23: Proceedings of the 31th ACM International Conference on Multimedia.*


---

![The structure of the proposed method](https://github.com/liuhw01/DUML/blob/main/checkpoint/pipleine.png)
Domain adaptation has driven the progress of Facial Expression Recognition (FER). Existing cross-domain FER methods focus only on the domain alignment of a single source domain to the target domain, ignoring the importance of multisource domains that contain richer knowledge. However, Cross-Multidomain FER (CMFER) needs to combat the domain conflicts caused by the uncertainty of intradomain annotations and the inconsistency of interdomain distributions. To this end, this paper proposes a Domain-Uncertain Mutual Learning (DUML) method to deal with the more challenging
CMFER problem. Specifically, we consider a domain-specific global perspective for domain-invariance representation and domain fusion for facial generic detail representation to mitigate cross- domain distribution differences. Further, we develop Intra-Domain Uncertainty (Intra-DU) and Inter-Domain Uncertainty (Inter-DU) to combat the large dataset shifts caused by annotation uncertainty. Finally, extensive experimental results on multiple benchmark across multidomain FER datasets demonstrate the remarkable effectiveness of DUML against CMFER uncertainty.

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

  
## If you use this work, please cite our paper

```
@inproceedings{10.1145/3581783.3611702,
author = {Liu, Hanwei and Cai, Huiling and Lin, Qingcheng and Li, Xuefeng and Xiao, Hui},
title = {Learning from More: Combating Uncertainty Cross-Multidomain for Facial Expression Recognition},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3611702},
doi = {10.1145/3581783.3611702},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {5889–5898},
numpages = {10},
keywords = {adversarial learning, cross multiple domains, facial expression recognition, negative transfer},
location = {Ottawa ON, Canada},
series = {MM '23}
}
