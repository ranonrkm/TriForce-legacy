<div align="center">
<h1><img src="imgs/triforce.png" height="40px" align="top"/> TriForce: Lossless Acceleration of Long Sequence <br> Generation with Hierarchical Speculative Decoding
</h1>


**trainig-free, accelerate long-context model inference**
</div>
<div align="center">
<b>Hanshi Sun</b><sup>1</sup>,
<b>Zhuoming Chen</b><sup>1</sup>,
<b>Xinyu Yang</b><sup>1</sup>,
<b>Yuandong Tian</b><sup>2</sup>,
<b>Beidi Chen</b><sup>1,2</sup>,
</div>

<div align="center">
<sup>1</sup>Carnegie Mellon University
<sup>2</sup>Meta AI (FAIR)
</div>

<br>
<img src="imgs/sys.png" align="top"/>


## Environment Setup
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


## Citation
If you find TriForce useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{sun2024triforce,
  title={TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding},
  author={Sun, Hanshi and Chen, Zhuoming and Yang, Xinyu and Tian, Yuandong and Chen, Beidi},
  journal={arXiv preprint arXiv:2404.XXXX},
  year={2024}
}
```
