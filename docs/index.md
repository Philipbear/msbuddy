# msbuddy

![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![docs](https://readthedocs.org/projects/msbuddy/badge/?version=latest)](https://msbuddy.readthedocs.io/en/latest/?badge=latest)

[**msbuddy**](https://github.com/Philipbear/msbuddy) is developed for molecular formula annotation in mass spectrometry-based small molecule analysis.

**msbuddy** is able to provide formula annotations for:
  * metabolic features with or without MS1 isotopic pattern 
  * metabolic features with or without MS/MS spectra
  * both positive and negative ionization modes

**msbuddy** enables candidate space shrinkage via bottom-up MS/MS interrogation, accurate formula annotation via machine learning and false discovery rate estimation.
Please refer to [our paper](https://doi.org/10.1038/s41592-023-01850-x) for more details.

Source codes are available at [GitHub](https://github.com/Philipbear/msbuddy).

```{toctree}
---
caption: Contents
maxdepth: 1
---

install
quickstart
paramset
datain
dataout
result
pyapi
cmdapi
contact
```
 
## Citation
When using **msbuddy**, please cite:
* Xing, S., Shen, S., Xu, B. et al. [BUDDY: molecular formula discovery via bottom-up MS/MS interrogation](https://doi.org/10.1038/s41592-023-01850-x). _Nat Methods_ **20**, 881–890 (2023).

## Contact
Shipei Xing: philipxsp@hotmail.com


## License
MIT License

Copyright 2023 Shipei Xing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.