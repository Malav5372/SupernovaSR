
# How to use the degradation model:
```python
from utils import utils_blindsr as blindsr
img_lq, img_hq = blindsr.degradation_SupernovaSR_plus(img, sf=4, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
```

