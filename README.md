# Image Style Transfer in TensorFlow

A tensorflow implementation of image style transfer described in the paper:
- [Image Style Transfer Using Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

The code runs in [Eager Execution](https://www.tensorflow.org/guide/eager).

## Usage

Download the VGG19 npy file from: [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) and put the npy file `vgg19.npy` in the `model` folder.


```bash
python train.py --content_path=<content_image_path> --style_path=<style_image_path> --output_path=<output_path>
```

Optional arguments:

- `--vgg19_npy_path`: VGG19 model path. Default value is `./model/vgg19.npy`.
- `--save_interval`: Save image interval. Default value is `10`.
- `--alpha`: Content weight. Default value is `0.0005`.
- `--beta`: Style weight. Default value is `1`.
- `--learning_rate`: Learning rate. Default value is `5`.
- `--optimizer`: Optimizer to use, only choose in `adam`, `sgd`, `momentum`. Default value is `adam`.

## Sample result
Content image  
<img src="sample/content.jpg" height="500px">

Output image  
<img src="sample/output.jpg" height="500px">













