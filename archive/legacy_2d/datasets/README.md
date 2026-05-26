# Downloading datasets / 下载数据集

This directory includes instructions and scripts for downloading ImageNet, LSUN bedrooms, and CIFAR-10 for use in this codebase.
本目录包含下载 ImageNet、LSUN bedrooms 和 CIFAR-10 的说明和脚本，供此代码库使用。

## ImageNet-64 / ImageNet-64

To download unconditional ImageNet-64, go to [this page on image-net.org](http://www.image-net.org/small/download.php) and click on "Train (64x64)". Simply download the file and unzip it, and use the resulting directory as the data directory (the `--data_dir` argument for the training script).
要下载无条件 ImageNet-64，请访问 [image-net.org 上的此页面](http://www.image-net.org/small/download.php)，点击 "Train (64x64)"。下载文件并解压，将解压后的目录用作数据目录（训练脚本的 `--data_dir` 参数）。

## Class-conditional ImageNet / 类别条件 ImageNet

For our class-conditional models, we use the official ILSVRC2012 dataset with manual center cropping and downsampling. To obtain this dataset, navigate to [this page on image-net.org](http://www.image-net.org/challenges/LSVRC/2012/downloads) and sign in (or create an account if you do not already have one). Then click on the link reading "Training images (Task 1 & 2)". This is a 138GB tar file containing 1000 sub-tar files, one per class.
对于类别条件模型，我们使用官方 ILSVRC2012 数据集，并进行手动中心裁剪和下采样。要获取此数据集，请访问 [image-net.org 上的此页面](http://www.image-net.org/challenges/LSVRC/2012/downloads) 并登录（如无账户则需创建）。然后点击 "Training images (Task 1 & 2)" 链接。这是一个 138GB 的 tar 文件，包含 1000 个子 tar 文件，每个类别一个。

Once the file is downloaded, extract it and look inside. You should see 1000 `.tar` files. You need to extract each of these, which may be impractical to do by hand on your operating system. To automate the process on a Unix-based system, you can `cd` into the directory and run this short shell script:
下载完成后，解压并查看内容。你会看到 1000 个 `.tar` 文件。你需要逐一解压每个文件，手工操作可能不太现实。在 Unix 系统上，可以 `cd` 到该目录并运行以下简短 shell 脚本来自动化处理：

```
for file in *.tar; do tar xf "$file"; rm "$file"; done
```

This will extract and remove each tar file in turn.
这将依次解压并删除每个 tar 文件。

Once all of the images have been extracted, the resulting directory should be usable as a data directory (the `--data_dir` argument for the training script). The filenames should all start with WNID (class ids) followed by underscores, like `n01440764_2708.JPEG`. Conveniently (but not by accident) this is how the automated data-loader expects to discover class labels.
解压所有图片后，生成的目录可用作数据目录（训练脚本的 `--data_dir` 参数）。文件名均应以 WNID（类别 ID）开头，后跟下划线，如 `n01440764_2708.JPEG`。方便的是（并非偶然），这正是自动数据加载器期望发现类别标签的方式。

## CIFAR-10 / CIFAR-10

For CIFAR-10, we created a script [cifar10.py](cifar10.py) that creates `cifar_train` and `cifar_test` directories. These directories contain files named like `truck_49997.png`, so that the class name is discernable to the data loader.
对于 CIFAR-10，我们创建了脚本 [cifar10.py](cifar10.py)，它会生成 `cifar_train` 和 `cifar_test` 目录。这些目录包含以 `truck_49997.png` 这样命名的文件，使数据加载器能够识别类别名称。

The `cifar_train` and `cifar_test` directories can be passed directly to the training scripts via the `--data_dir` argument.
`cifar_train` 和 `cifar_test` 目录可直接通过 `--data_dir` 参数传递给训练脚本。

## LSUN bedroom / LSUN bedroom

To download and pre-process LSUN bedroom, clone [fyu/lsun](https://github.com/fyu/lsun) on GitHub and run their download script `python3 download.py bedroom`. The result will be an "lmdb" database named like `bedroom_train_lmdb`. You can pass this to our [lsun_bedroom.py](lsun_bedroom.py) script like so:
要下载和预处理 LSUN bedroom，请在 GitHub 上克隆 [fyu/lsun](https://github.com/fyu/lsun) 并运行其下载脚本 `python3 download.py bedroom`。结果将是一个名为 `bedroom_train_lmdb` 的 "lmdb" 数据库。可以按如下方式将其传递给我们的 [lsun_bedroom.py](lsun_bedroom.py) 脚本：

```
python lsun_bedroom.py bedroom_train_lmdb lsun_train_output_dir
```

This creates a directory called `lsun_train_output_dir`. This directory can be passed to the training scripts via the `--data_dir` argument.
这将创建一个名为 `lsun_train_output_dir` 的目录。该目录可通过 `--data_dir` 参数传递给训练脚本。
