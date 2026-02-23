# SONA
Stochastic directional Oversampling using Negative Anomalous scores for imbalanced dataset

## Installation
To install `SONA`, type the following command in the terminal

```
pip install sona-oversampling            # normal install
pip install --upgrade sona-oversampling  # or update if needed
```

**Required Dependencies** :
- Python 3.9 or higher
- numpy>=2.0.2
- scipy>=1.8.1

## SONA
The SONA function generates synthetic samples for the minority class by identifying border areas between classes and creating new points within a calculated safety radius.

### SONA(X,y, min_label, new_label = 0)
|Parameter  |Type           |Description |
|---        |---            |---           |
|X          |numpy.ndarray  |The input feature matrix of shape $(n_{samples}, n_{features})$.|
|y          |numpy.ndarray  |The target labels.|
|min_label  |int            |The specific label identified as the minority class.|
|new_label  |int            |An offset added to min_label for the new synthetic samples (default is 0).|

**Returns**
|Parameter  |Type           |Description |
|---        |---            |---           |
|X_augmented| numpy.ndarray |containing the original features plus the new synthetic samples.|
|y_augmented| numpy.ndarray | containing the original labels plus labels for the synthetic samples.|

## Usage Example
```
from sona_oversampling import SONA
from sklearn.datasets import make_circles, make_moons, make_blobs

# Generate 'Double circle' dataset with imbalance
X_circles, y_circles = make_circles(n_samples=(500, 100), noise=0.05, random_state=42)

# Generate 'Blue-moons' dataset with imbalance
X_moons, y_moons = make_moons(n_samples=(500, 100), noise=0.1, random_state=42)

# Generate imbalanced 'Gaussian cluster' dataset
X_blobs, y_blobs = make_blobs(n_samples=[500, 50], cluster_std=0.5, centers=[[0, 0], [1, 1]], random_state=42)

synthetic_datasets = [
    ("Double circle", (X_circles, y_circles)),
    ("Blue-moons", (X_moons, y_moons)),
    ("Gaussian cluster", (X_blobs, y_blobs))
]

for name, (X_original, y_original) in synthetic_datasets:
  
  X_oversampled, y_oversampled = SONA(X_original, y_original, min_label= 1, new_label=1)
  mask_maj = (y_oversampled == 0)
  mask_min_orig = (y_oversampled == 1)
  mask_min_syn = (y_oversampled == 2)

  plt.figure(figsize=(10, 8))

  # Plot Majority Class
  plt.scatter(X_oversampled[mask_maj, 0], X_oversampled[mask_maj, 1],
              c='grey', label='Majority Class (0)', alpha=0.5, s=20)

  # Plot Original Minority Class
  plt.scatter(X_oversampled[mask_min_orig, 0], X_oversampled[mask_min_orig, 1],
              c='blue', label='Original Minority (1)', s=30, edgecolors='k')

  # Plot Synthetic Minority Class
  plt.scatter(X_oversampled[mask_min_syn, 0], X_oversampled[mask_min_syn, 1],
              c='red', label='Synthetic Samples (2)', marker='x', s=40, alpha=0.8)

  plt.title(f"SONA Oversampling: {name} Dataset", fontsize=14)
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.legend(loc='best')
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.axis('equal')

  plt.show()
```
**Output**
![Double circles](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_circle.png?raw=true)
![Blue moons](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_blue_moons.png?raw=true)
![Gaussian clusters](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_gaussian_cluster.png?raw=true)