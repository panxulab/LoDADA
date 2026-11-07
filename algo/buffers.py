"""Replay buffers and the LoDADA source-data filter.

The filter follows Algorithm 1 of the paper:
1. cluster target transitions by next state s' with KMeans;
2. assign each source transition to its nearest target centroid if it lies within
   `diameter_threshold` x (mean target cluster radius);
3. in every cluster, train a source-vs-target classifier on (s, a) and estimate the
   cluster-level KL divergence and point-wise KL estimates;
4. rank clusters by KL, split them into three tiers, and keep the top
   (filter_ratio_small, filter_ratio_medium, filter_ratio_large) fraction of source
   samples (ranked by point-wise KL) in each tier.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer(object):
    """Plain buffer used for the target-domain data."""

    def __init__(self, device):
        self.device = device
        self.size = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def convert_D4RL(self, tar_dataset):
        self.state = tar_dataset['observations']
        self.action = tar_dataset['actions']
        self.next_state = tar_dataset['next_observations']
        self.reward = tar_dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - tar_dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_local_tar(self, tar_dataset, num_samples=5000, seed=0):
        """Localized-perturbation setting: the target data is a random subset of
        `num_samples` transitions from the unperturbed D4RL dataset."""
        np.random.seed(seed)

        N = tar_dataset['observations'].shape[0]
        n_samples = min(num_samples, N)
        indices = np.random.choice(N, size=n_samples, replace=False)

        self.state = tar_dataset['observations'][indices]
        self.action = tar_dataset['actions'][indices]
        self.next_state = tar_dataset['next_observations'][indices]
        self.reward = tar_dataset['rewards'][indices].reshape(-1, 1)
        self.not_done = 1. - tar_dataset['terminals'][indices].reshape(-1, 1)
        self.size = self.state.shape[0]


class LoDADAReplayBuffer(object):
    """Source-domain buffer. Stores the filtered source data and the point-wise KL
    estimate (`divergence`) of every kept transition."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.size = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.divergence[ind]).to(self.device),
        )

    def convert_D4RL(self, src_dataset, tar_dataset):
        """Filter the source dataset against the target dataset and store the result."""
        (self.state, self.action, self.next_state, self.reward,
         self.not_done, self.divergence) = self.filter(
            src_dataset, tar_dataset,
            num_clusters=self.config['num_clusters'],
            input_noise=self.config['sigma'],
        )
        self.size = self.state.shape[0]

    def convert_D4RL_local(self, src_dataset, tar_dataset, shift_levels, num_regions):
        """Localized-perturbation setting: perturb the source dataset region by region,
        then filter it against the target dataset."""
        s_modified, a_all, sp_all = self.make_local_difference(src_dataset, shift_levels, num_regions)

        self.state = np.array(s_modified, dtype=np.float32)
        self.action = np.array(a_all, dtype=np.float32)
        self.next_state = np.array(sp_all, dtype=np.float32)
        self.not_done = np.array(1. - src_dataset['terminals'].reshape(-1, 1), dtype=np.float32)
        self.reward = np.array(src_dataset['rewards'].reshape(-1, 1), dtype=np.float32)

        (self.state, self.action, self.next_state, self.reward,
         self.not_done, self.divergence) = self.filter(
            {
                "observations": self.state,
                "actions": self.action,
                "next_observations": self.next_state,
                "rewards": self.reward,
                "terminals": src_dataset["terminals"],
            },
            tar_dataset,
            num_clusters=self.config['num_clusters'],
            input_noise=0,
        )
        self.not_done = np.array(self.not_done, dtype=np.float32)
        self.size = self.state.shape[0]

    @staticmethod
    def make_local_difference(dataset, shift_levels, num_regions, seed=0):
        """Split the dataset into `num_regions` KMeans regions (on s') and add Gaussian
        noise with standard deviation `shift_levels[g]` to the states s of the regions
        in group g. Regions are shuffled and split into len(shift_levels) equal groups."""
        np.random.seed(seed)
        s_all = dataset["observations"]
        a_all = dataset["actions"]
        sp_all = dataset["next_observations"]
        num_regions = int(num_regions)

        kmeans = KMeans(n_clusters=num_regions, random_state=0, n_init=10)
        region_labels = kmeans.fit_predict(sp_all)

        all_regions = np.arange(num_regions)
        np.random.shuffle(all_regions)

        n = num_regions
        num_groups = len(shift_levels)
        groups = [all_regions[g * n // num_groups:(g + 1) * n // num_groups] for g in range(num_groups)]

        s_modified = s_all.copy()
        for regions, shift_std in zip(groups, shift_levels):
            for rid in regions:
                idx = np.where(region_labels == rid)[0]
                noise = np.random.normal(0, float(shift_std), size=sp_all[idx].shape)
                s_modified[idx] += noise

        return s_modified, a_all, sp_all

    def filter(self, src_dataset, tar_dataset, num_clusters, input_noise):
        self.num_clusters = num_clusters
        s_src, a_src, sp_src, r_src = (src_dataset["observations"], src_dataset["actions"],
                                       src_dataset["next_observations"], src_dataset["rewards"])
        s_tar, a_tar, sp_tar, r_tar = (tar_dataset["observations"], tar_dataset["actions"],
                                       tar_dataset["next_observations"], tar_dataset["rewards"])

        # --- 1. KMeans on target next states ---
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0,
                        n_init=self.config['kmeans_n_init'])
        target_cluster_labels = kmeans.fit_predict(sp_tar)
        centroids = kmeans.cluster_centers_

        clusters = {i: [] for i in range(self.num_clusters)}
        cluster_domain_labels = {i: [] for i in range(self.num_clusters)}
        cluster_sp = {i: [] for i in range(self.num_clusters)}

        for idx, label in enumerate(target_cluster_labels):
            clusters[label].append((s_tar[idx], a_tar[idx], sp_tar[idx], r_tar[idx], tar_dataset['terminals'][idx]))
            cluster_domain_labels[label].append(1)
            cluster_sp[label].append(sp_tar[idx])

        # --- 2. Diameter threshold: delta x mean radius of the target clusters ---
        target_cluster_radii = []
        for i in range(self.num_clusters):
            sp_points = np.array(cluster_sp[i])
            diameter = np.max(pdist(sp_points)) if len(sp_points) > 1 else 0.0
            target_cluster_radii.append(0.5 * diameter)
        diameter_threshold = self.config['diameter_threshold'] * np.mean(target_cluster_radii)

        # Assign source samples to the nearest centroid; drop those outside the threshold
        distances = cdist(sp_src, centroids)
        nearest_distances = np.min(distances, axis=1)
        nearest_centroids = np.argmin(distances, axis=1)
        within_diameter_mask = nearest_distances <= diameter_threshold
        print(f"Removed {np.sum(~within_diameter_mask)} source samples outside the diameter threshold")

        s_src_filtered = s_src[within_diameter_mask]
        a_src_filtered = a_src[within_diameter_mask]
        sp_src_filtered = sp_src[within_diameter_mask]
        r_src_filtered = r_src[within_diameter_mask]
        d_src_filtered = src_dataset['terminals'][within_diameter_mask]
        nearest_centroids_filtered = nearest_centroids[within_diameter_mask]

        for i in range(self.num_clusters):
            source_indices = np.where(nearest_centroids_filtered == i)[0]
            for idx_src in source_indices:
                clusters[i].append((s_src_filtered[idx_src], a_src_filtered[idx_src], sp_src_filtered[idx_src],
                                    r_src_filtered[idx_src], d_src_filtered[idx_src]))
                cluster_domain_labels[i].append(0)
                cluster_sp[i].append(sp_src_filtered[idx_src])

        for i in range(self.num_clusters):
            domain_arr = np.array(cluster_domain_labels[i])
            print(f"Cluster {i}: {np.sum(domain_arr == 0)} source / {np.sum(domain_arr == 1)} target samples")

        # --- 3. Per-cluster domain classifier ---
        def train_classifier_for_neighborhood(s_src_n_train, a_src_n_train, s_tar_n_train, a_tar_n_train,
                                              s_src_n_test, a_src_n_test, s_tar_n_test, a_tar_n_test,
                                              input_noise, epochs, batch_size=32, patience=15):
            input_dim = s_src_n_train.shape[1] + a_src_n_train.shape[1]
            model = BinaryClassifier(input_dim=input_dim, input_noise=input_noise)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.BCELoss()

            z_src_test = np.concatenate([s_src_n_test, a_src_n_test], axis=1)
            z_tar_test = np.concatenate([s_tar_n_test, a_tar_n_test], axis=1)
            X_test = np.concatenate([z_src_test, z_tar_test], axis=0)
            y_test = np.concatenate([np.zeros(len(z_src_test)), np.ones(len(z_tar_test))])
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            best_loss = float("inf")
            best_model_state = None
            patience_counter = 0

            for epoch in range(epochs):
                model.train()

                # Resample to balance the two classes
                z_src_train = np.concatenate([s_src_n_train, a_src_n_train], axis=1)
                z_tar_train = np.concatenate([s_tar_n_train, a_tar_n_train], axis=1)
                min_len = min(len(z_src_train), len(z_tar_train))
                z_src_sampled = resample(z_src_train, n_samples=min_len)
                z_tar_sampled = resample(z_tar_train, n_samples=min_len)

                X_train = np.concatenate([z_src_sampled, z_tar_sampled], axis=0)
                y_train = np.concatenate([np.zeros(min_len), np.ones(min_len)])
                X_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

                loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
                for xb, yb in loader:
                    preds = model(xb)
                    loss = -(yb * torch.log(preds + 1e-8) + (1 - yb) * torch.log(1 - preds + 1e-8)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    preds = model(X_test_tensor)
                    val_loss = loss_fn(preds, y_test_tensor).item()
                    acc = ((preds > 0.5).float() == y_test_tensor).float().mean().item()

                # Early stopping
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            return model, acc

        models = {}
        for n in range(self.num_clusters):
            s_n, a_n, domain_labels_n = self._cluster_arrays(clusters[n], cluster_domain_labels[n])
            domain_mask_n = (domain_labels_n == 1)
            s_src_n, a_src_n = s_n[~domain_mask_n], a_n[~domain_mask_n]
            s_tar_n, a_tar_n = s_n[domain_mask_n], a_n[domain_mask_n]

            if len(s_src_n) < 2 or len(s_tar_n) < 2:
                print(f"Skipping cluster {n} (insufficient data).")
                continue

            s_src_train, s_src_test, a_src_train, a_src_test = train_test_split(s_src_n, a_src_n, test_size=0.3, random_state=42)
            s_tar_train, s_tar_test, a_tar_train, a_tar_test = train_test_split(s_tar_n, a_tar_n, test_size=0.3, random_state=42)

            model_n, acc_n = train_classifier_for_neighborhood(s_src_train, a_src_train, s_tar_train, a_tar_train,
                                                               s_src_test, a_src_test, s_tar_test, a_tar_test, input_noise,
                                                               epochs=self.config['classifier_epochs'])
            models[n] = model_n
            print(f"Cluster {n}: classifier test accuracy {acc_n:.4f}")

        # --- KL estimates from the classifier (density-ratio trick) ---
        def estimate_kl(model, s_src_n, a_src_n, s_tar_n, a_tar_n):
            model.eval()
            with torch.no_grad():
                z_src = np.concatenate([s_src_n, a_src_n], axis=1)
                N0, N1 = len(z_src), len(s_tar_n)
                tar_frac = N1 / (N0 + N1)  # fraction of target samples in the cluster
                Dn_z = model(torch.tensor(z_src, dtype=torch.float32)).squeeze().numpy()
                Dn_z = np.clip(Dn_z, 1e-6, 1 - 1e-6)
                w_hat = ((1 - tar_frac) * Dn_z) / (tar_frac * (1 - Dn_z))
                kl_per_point = -np.log(w_hat)
            return kl_per_point, np.mean(kl_per_point)

        kl_results = []
        for n in models:
            s_n, a_n, domain_labels_n = self._cluster_arrays(clusters[n], cluster_domain_labels[n])
            domain_mask_n = (domain_labels_n == 1)
            s_src_n, a_src_n = s_n[~domain_mask_n], a_n[~domain_mask_n]
            s_tar_n, a_tar_n = s_n[domain_mask_n], a_n[domain_mask_n]
            kl_point, kl_neigh = estimate_kl(models[n], s_src_n, a_src_n, s_tar_n, a_tar_n)
            kl_results.append((n, kl_neigh, kl_point))

        # --- 4. Three KL tiers and tier-wise keep ratios ---
        sorted_results = sorted(kl_results, key=lambda x: x[1])
        third = len(sorted_results) // 3
        small_kl_clusters = [cid for cid, _, _ in sorted_results[:third]]
        medium_kl_clusters = [cid for cid, _, _ in sorted_results[third:2 * third]]
        print("Small KL clusters:", small_kl_clusters)
        print("Medium KL clusters:", medium_kl_clusters)
        print("Large KL clusters:", [cid for cid, _, _ in sorted_results[2 * third:]])

        kept_states, kept_actions, kept_next_states, kept_rewards, kept_not_done, kept_divergences = [], [], [], [], [], []
        for n, kl_val, kl_point in sorted_results:
            data_n = clusters[n]
            domain_labels_n = np.array(cluster_domain_labels[n])
            s_n = np.array([x[0] for x in data_n])
            a_n = np.array([x[1] for x in data_n])
            s_next_n = np.array([x[2] for x in data_n])
            r_n = np.array([x[3] for x in data_n])
            done_n = np.array([x[4] for x in data_n])

            mask_src = (domain_labels_n == 0)
            s_c, a_c, s_next_c, r_c, done_c = s_n[mask_src], a_n[mask_src], s_next_n[mask_src], r_n[mask_src], done_n[mask_src]

            if n in small_kl_clusters:
                keep_ratio = self.config['filter_ratio_small']
            elif n in medium_kl_clusters:
                keep_ratio = self.config['filter_ratio_medium']
            else:
                keep_ratio = self.config['filter_ratio_large']

            # keep the samples with the smallest point-wise KL estimates
            kl_points = np.array(kl_point)
            idx_sorted = np.argsort(kl_points)
            idx_keep = idx_sorted[:int(len(idx_sorted) * keep_ratio)]

            kept_states.append(s_c[idx_keep])
            kept_actions.append(a_c[idx_keep])
            kept_next_states.append(s_next_c[idx_keep])
            kept_rewards.append(r_c[idx_keep].reshape(-1, 1))
            kept_not_done.append((1. - done_c[idx_keep]).reshape(-1, 1))
            kept_divergences.append(kl_points[idx_keep].reshape(-1, 1))

        kept_states = np.concatenate(kept_states, axis=0)
        kept_actions = np.concatenate(kept_actions, axis=0)
        kept_next_states = np.concatenate(kept_next_states, axis=0)
        kept_rewards = np.concatenate(kept_rewards, axis=0)
        kept_not_done = np.concatenate(kept_not_done, axis=0)
        kept_divergences = np.concatenate(kept_divergences, axis=0)

        print("Final kept source size:", kept_states.shape[0])
        return kept_states, kept_actions, kept_next_states, kept_rewards, kept_not_done, kept_divergences

    @staticmethod
    def _cluster_arrays(data_n, domain_labels_n):
        s_n = np.array([x[0] for x in data_n])
        a_n = np.array([x[1] for x in data_n])
        return s_n, a_n, np.array(domain_labels_n)


class GaussianNoise(nn.Module):
    """Adds Gaussian input noise to the classifier during training only."""

    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            return x + torch.randn_like(x) * self.stddev
        return x


class BinaryClassifier(nn.Module):
    """Source (label 0) vs. target (label 1) classifier on (s, a)."""

    def __init__(self, input_dim, input_noise):
        super().__init__()
        self.noise = GaussianNoise(input_noise)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(self.noise(x))
