# Reproducing Tile2Vec: learning from satellite images without labels

**CMPE 257 · San José State University · Jaafar Ben Khaled**

This project reproduces **Tile2Vec** (Jean et al., AAAI 2019) on NAIP aerial images of California's Central Valley. Then I go a bit further: a stronger baseline, paired significance tests, a test of how few labels you can get away with, and my own model trained from scratch.

---

## The problem

A CNN learns from labeled examples. In the forward step it sees an image and predicts a score for each class, the loss compares that prediction with the label, and backpropagation adjusts the weights. That works really well, but it needs thousands of labeled images, and with too few it just memorizes them. That's overfitting.

For normal photos, labeling is easy. Anyone can say "this is a cat" or "this is a car." Satellite images are the opposite. The images are free and everywhere (Sentinel-2 photographs the whole planet about every 5 days), but **labeling them is hard**. From above, grapes and almonds look almost the same, so you need an expert or someone who actually goes to the field. Even the labels in this project come from a USDA crop map that only exists because the US government surveys farmers every year.

So the question is: **can a CNN learn useful features from satellite images without any labels at all?**

I care about this for a personal reason. My family has an olive farm in Tunisia. There are years of free satellite images of it, and almost no labels.

## The idea: location instead of labels

Tile2Vec replaces labels with **location**. A field doesn't stop after 30 meters, so a tile next to a vineyard is very likely vineyard too. Geographers call this Tobler's first law: *near things are more related than distant things*.

So one training example is a **triplet** of three tiles:

- an **anchor**: a random 50×50 tile,
- a **neighbor**: sampled close to it, so probably the same land,
- a **distant** tile: sampled far away, so probably different.

No labels anywhere. It's not always true (a distant tile can be the same crop by chance), but over 100,000 triplets it's true often enough.

Tile2Vec is not a new network. It's a new **way of training** one.

## The network: TileNet

Tile2Vec trains a **TileNet**, which is a ResNet-18 adapted for small satellite tiles. A ResNet is a CNN built from **residual blocks**: two convolutions, plus a **shortcut** that adds the block's input back to its output. That shortcut gives the gradient a direct path back, which keeps deep networks trainable.

TileNet changes four things compared to a normal ResNet-18:

1. **4 bands in, not 3:** red, green, blue, plus near-infrared, which shows healthy vegetation really well.
2. **No early shrinking.** A normal ResNet starts with a 7×7 convolution with stride 2 and a max pool, which shrinks a 224-pixel photo 4 times right away. Our tiles are only 50 pixels, so TileNet uses a small 3×3 convolution with stride 1 and no max pool. The maps stay 50×50.
3. **A 5th group of residual blocks.** Since it doesn't shrink at the start, it needs one more group to bring the maps down.
4. **No output layer, no softmax.** It stops at 512 numbers.

Here's the path a tile takes:

| Step | What happens | Output |
|---|---|---|
| Input | the tile | 4 grids of 50×50 |
| First conv | 64 filters of 3×3, stride 1 | 64 maps of 50×50 |
| Group 1 | 2 residual blocks | 64 maps of 50×50 |
| Group 2 | 2 residual blocks, the first with stride 2 | 128 maps of 25×25 |
| Group 3 | same | 256 maps of 13×13 |
| Group 4 | same | 512 maps of 7×7 |
| Group 5 | same | 512 maps of 4×4 |
| Average pool | each map becomes 1 number | **512 numbers** |

The way I think about it: as the maps shrink, each 3×3 window covers a bigger part of the tile, so the early layers see small details and the last ones see the whole field. At the same time the number of filters grows, so the network can detect more kinds of those bigger patterns. It trades "where" for "what."

Those final 512 numbers are the tile's **embedding**: its point in a 512-dimensional space.

## Training: the triplet loss

All three tiles go through the **same** TileNet, with the same weights, and each becomes 512 numbers. Then we measure two distances, both from the anchor:

- **d(a, n):** anchor to neighbor. We want this small.
- **d(a, d):** anchor to distant. We want this big.

The easiest way to picture the loss is a number line. Put the anchor at 0, then the neighbor, then an orange "margin zone" of length 10 starting at the neighbor. **The rule is that the distant tile has to be past that zone.** If it's still inside, the loss is how far it's missing. Once it's past, the loss is 0.

```
loss = max(0, d(a,n) − d(a,d) + 10) + 0.01 × (size of the three embeddings)
```

Example: neighbor at 6.2 and distant at 9.1 gives 6.2 − 9.1 + 10 = **7.1**.

The small second term is an **L2 penalty**. Without it, the model could cheat by multiplying every embedding by 100: every distance grows, the loss drops, and nothing was learned. The penalty makes that kind of stretching cost something, so the only way to lower the loss is to actually put similar tiles close together.

Then it's a normal training loop. Backpropagation sends the error back through the three passes, and since they share the same weights, their gradients add up. Adam updates the weights. That's one batch of 50 triplets. 2,000 batches make one epoch, which is all 100,000 triplets. Tiles are randomly flipped and rotated each time, because from above a field is the same field whichever way you turn it.

## How I evaluated it

Once TileNet is trained, it's **frozen**. It just turns each tile into 512 numbers. If it learned well, tiles of the same crop end up close together, so a simple classifier only has to draw boundaries between the groups. That needs very few labels.

To check that, I keep the classifier the same and change only the features. Three kinds:

- **Tile2Vec:** the 512 numbers from the authors' pretrained TileNet, all 4 bands, learned with **no labels**.
- **ImageNet ResNet-18:** the standard "off the shelf" choice, a ResNet trained on 1.2 million labeled everyday photos. It only knows color photos, so it gets red, green and blue, resized from 50 to 224 pixels.
- **Raw pixels + PCA:** no network at all. The 10,000 pixel values of the tile, compressed to 10 numbers. This is the floor.

On top of each one I train the same two classifiers: a **random forest** (100 trees, can draw curvy boundaries) and a **logistic regression** (only straight boundaries, so it directly tests how clean the groups are).

I use the 1,000 labeled tiles: **800 to train the classifier, 200 to test**, repeated over **100 different random splits**. One split can be lucky, and scores move by about 3 points from split to split. Every method gets exactly the same 100 splits.

Two scores:

- **Accuracy:** the share of test tiles predicted correctly. Always guessing "grapes" already gets about 35%, because 354 of the 1,000 tiles are grapes.
- **Macro F1:** a score per crop, then averaged, so rare crops count as much as grapes.

## Results

With 800 training labels (notebook `04_baselines.ipynb`):

| Features | RF accuracy | LR accuracy | RF macro F1 | LR macro F1 |
|---|---|---|---|---|
| **Tile2Vec (pretrained)** | **0.717** | **0.703** | **0.393** | **0.410** |
| ImageNet ResNet-18 (RGB) | 0.685 | 0.685 | 0.227 | 0.351 |
| Raw pixels + PCA(10) | 0.623 | 0.576 | 0.247 | 0.165 |
| Always "grapes" | 0.358 | | | |

Why this order:

- **Raw pixels are last** because there's no learning at all.
- **ImageNet does better.** It never saw a satellite image, but its first filters (edges, textures, colors) work on any image.
- **Tile2Vec does best** for three reasons: it learned from the same kind of images (farmland seen from above), it uses near-infrared, and it was trained so that similar land ends up close together, which is exactly what a crop classifier needs.

**Is the gap real?** Since every method used the same 100 splits, I compared Tile2Vec and ImageNet split by split with a **paired Wilcoxon test**. For each split it looks at which method won and by how much, and checks whether the wins are clearly on one side or could just be luck.

- Random forest: Tile2Vec wins **89 of 100 splits**, +3.2 points on average, p = 6.8 × 10⁻¹⁶
- Logistic regression: Tile2Vec wins **69 of 100 splits**, +1.7 points, p = 2.6 × 10⁻⁷

### What if you only have a few labels?

The whole promise of Tile2Vec is "few labels are enough," so I hid most of the labels. Same 200 test tiles, but the classifier only learns from 50, 100, 200, 400 or 800 of them, as if the rest had never been labeled.

| Labels | Tile2Vec | ImageNet | Raw pixels |
|---|---|---|---|
| 50 | 0.542 | **0.571** | 0.464 |
| 100 | 0.604 | **0.618** | 0.508 |
| 200 | **0.650** | 0.646 | 0.551 |
| 400 | **0.686** | 0.669 | 0.595 |
| 800 | **0.727** | 0.689 | 0.631 |

(Random forest, 20 splits per point.)

From about **200 labels** up, Tile2Vec is on top. Below that, **the lines cross** and ImageNet wins, and with 100 paired splits that difference is significant (at 50 labels: 0.540 vs 0.581, p = 5.6 × 10⁻¹²).

My explanation: with 50 labels, both methods struggle, because most of the 28 crops don't even have one example. ImageNet drops less, most likely because its features are more general (learned from over a million labeled photos), while Tile2Vec's are more specialized for farmland and need a few hundred examples before that pays off.

**So it's a trade-off.** Features learned without any labels, from satellite tiles with near-infrared, beat features learned from 1.2 million labeled photos, as long as you have a few hundred labels for the classifier. With only dozens, the general ImageNet features are safer.

## Training my own TileNet

The results above use the authors' pretrained model. That checks my evaluation pipeline. To check that the **training** itself reproduces, I trained my own TileNet from scratch with the authors' code and their exact settings: margin 10, L2 0.01, Adam (lr 0.001), batches of 50, flips and rotations.

Each epoch takes about 14 minutes on a Colab T4, and the free GPU limit stopped me at **epoch 21 of 50**. Even so:

- the loss went from 4.79 to 3.22 and was still going down,
- accuracy passed ImageNet around epoch 5 and matched the authors' model from epoch 10,
- at epoch 21, on the same 100 splits: **0.722 vs 0.717** (random forest) and **0.714 vs 0.703** (logistic regression),
- macro F1 is a bit lower (0.367 vs 0.393), so it's slightly worse on rare crops.

So I'd say my model **matches** theirs rather than beats it. Training optimizes the triplet loss, not crop accuracy, and accuracy flattens after about 10 epochs. The extra half point can easily come from random differences between two training runs.

**In progress:** a second run that keeps 5,000 triplets aside as a validation set and logs training loss, validation loss and test accuracy every epoch, to check for overfitting. This was suggested during the final presentation.

## How my setup differs from the paper

| | The paper | This project |
|---|---|---|
| Labeled tiles | 1,000 and 10,000, from a 66-class area fully covered by the USDA map | The 1,000 tiles the authors released on GitHub, 28 classes |
| Tile2Vec accuracy (RF) | ~53% | ~72% |
| Pretrained ResNet baseline | trained on CIFAR-10 (50,000 tiny 32-pixel photos, 10 classes) | trained on ImageNet (1.2M photos, 1,000 classes), a stronger baseline |
| Repeats | 10 trials | 100 paired splits + Wilcoxon test |
| Scores | accuracy | accuracy + macro F1 |
| Fewest labels tested | 1,000 | 50 |

My accuracy is higher than the paper's because it's a different, easier set (28 classes instead of 66), so the two numbers can't be compared directly. What **does** reproduce is the ranking: Tile2Vec beats the pretrained ResNet and raw pixels, and my own trained model matches the authors'.

## Repository

| Notebook | What it does |
|---|---|
| `01_explore_data.ipynb` | Clones the authors' code, unzips the 1,000 labeled tiles, looks at the bands and labels, downloads the pretrained model, and does a first quick random forest check |
| `02_train_tile2vec.ipynb` | Builds and caches the three feature types, tests the low-label crossover with 100 paired splits, and trains my own TileNet (patches, checkpoints, resume) |
| `03_evaluate.ipynb` | Evaluates my own checkpoints: training loss plot, accuracy vs epochs, my model vs the pretrained one |
| `04_baselines.ipynb` | Main results table, Wilcoxon tests and the label curve |

## How to run it

Everything runs in **Google Colab**. Only training needs a GPU; the evaluation runs fine on CPU.

1. Open a notebook from this repo in Colab (each one has an "Open in Colab" badge).
2. The notebooks clone the authors' repo (`ermongroup/tile2vec`), unzip the 1,000 labeled tiles from it, and download the pretrained NAIP model (the Dropbox link is in the authors' README).
3. For training, you need the 100,000 NAIP triplets released by the authors. I keep them as `triplets.zip` in Google Drive under `MyDrive/tile2vec_data/`.
4. Run in order: `01` → `02` → `03` → `04`.

Features are cached in `MyDrive/tile2vec_data/features.npz`, and checkpoints are saved to `MyDrive/tile2vec_data/checkpoints/` after every epoch, so if Colab disconnects, rerunning the cell picks up where it stopped.

**Libraries:** PyTorch, torchvision, scikit-learn, SciPy, NumPy, Matplotlib (all preinstalled on Colab).

**Two small fixes to the 2018 code.** The math is unchanged; they're only needed so it runs on today's libraries:

- `import gdal` no longer works, and it's only used to read GeoTIFFs (never during training), so I replace it with an empty module.
- `loss.data[0]` is old PyTorch syntax, so I replace it with `loss.item()`.

## Limitations

- The 100 splits reuse the same 1,000 tiles, so they're not fully independent and the p-values are a bit optimistic. Winning 89% of splits is convincing anyway.
- Splits are random, not stratified, because some crops have only one tile.
- The ImageNet baseline only sees red, green and blue. That's how it's normally used, but it can't use near-infrared.
- I didn't train an RGB-only Tile2Vec, so I can't say how much of its advantage comes from near-infrared versus from learning on farmland images. That's a good next experiment.
- My own model is a single training run, stopped at epoch 21.

## What's next

Apply Tile2Vec to **Sentinel-2** images of olive groves in Tunisia, where labels basically don't exist. Most of the method stays the same: the input layer changes to Sentinel-2's 13 bands, and the tiles get bigger because the pixels are coarser (10 m instead of 0.6 m).

## Credits

- Neal Jean, Sherrie Wang, Anshul Samar, George Azzari, David Lobell, Stefano Ermon. *Tile2Vec: Unsupervised Representation Learning for Spatially Distributed Data.* AAAI 2019.
- Original code and data: [ermongroup/tile2vec](https://github.com/ermongroup/tile2vec) (MIT license).
- Labels: USDA Cropland Data Layer. Imagery: USDA NAIP.
