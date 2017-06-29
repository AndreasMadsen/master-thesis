# Master Thesis
**Semi-Supervised Neural Machine Translation - _for small bilingual datasets_**

_By: Andreas Madsen (June 2017)_

## Download

```shell
git clone https://github.com/AndreasMadsen/master-thesis.git
```

## Code

### Dependencies

The code was written using:
* Python 3.6
* TensorFlow 1.1
* Sugartensor. Some of my PRs haven't been merged yet, for now use: https://github.com/AndreasMadsen/sugartensor/tree/master-thesis
* tqdm
* numpy
* scipy
* R and ggplot2


### Run code

All the experiments ( and may more :o ) are in the `code/script` directory,
the plot generation code is in `code/plot`. The `jobs` and `grid` directory
is for running on the DTU LFS queue system.

### Dataset

The datasets are automatically downloaded, primarily Europarl v7 and
WMT NewsTest are used.

### Code License

The code license is MIT and is seperate from the main thesis license.

> Copyright (c) 2017 Andreas Madsen
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Report

The report is build using `make report`.

The figures are available as Incscape files. The Incscape files uses
`Modern Latin Roman` as the font, so make sure to have that installed.

### Report License

While the LaTeX code is available it is not open source. That means you are not
allowed to redistribute or modify the PDF, the LaTeX code, or any other file.
But you can redistribute this url:
https://github.com/AndreasMadsen/master-thesis
