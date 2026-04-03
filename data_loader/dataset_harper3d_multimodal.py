import numpy as np
from data_loader.dataset_harper3d import DatasetHarper3D


class DatasetHarper3D_multi(DatasetHarper3D):
    """
    Multi-modal version of Harper3D dataset for evaluation.
    """

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 data_path='./data/harper3d', include_spot=True, fps='30hz', **kwargs):
        self.multimodal_path = kwargs.get('multimodal_path', None)
        self.data_candi_path = kwargs.get('data_candi_path', None)
        super().__init__(mode, t_his, t_pred, actions, use_vel, data_path, include_spot, fps)

    def sample(self, n_modality=5):
        """Sample with multimodal ground truth."""
        subject = np.random.choice(self.subjects)
        if subject not in self.data:
            subject = list(self.data.keys())[0]
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        
        # For now, return None for multimodal (can be extended later)
        return traj[None, ...], None

    def sampling_generator(self, num_samples=1000, batch_size=8, n_modality=5):
        for i in range(num_samples // batch_size):
            sample = []
            sample_multi = []
            for j in range(batch_size):
                sample_i, sample_multi_i = self.sample(n_modality=n_modality)
                sample.append(sample_i)
                if sample_multi_i is not None:
                    sample_multi.append(sample_multi_i[None, ...])
            sample = np.concatenate(sample, axis=0)
            if len(sample_multi) > 0:
                sample_multi = np.concatenate(sample_multi, axis=0)
            else:
                sample_multi = None
            yield sample, sample_multi

    def iter_generator(self, step=25, n_modality=10):
        """Iterate over all sequences."""
        for sub in self.data.keys():
            data_s = self.data[sub]
            for act in data_s.keys():
                seq = data_s[act]
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj, None


if __name__ == '__main__':
    np.random.seed(0)
    dataset = DatasetHarper3D_multi('test', t_his=25, t_pred=100, data_path='./data/harper3d')
    print(f"Dataset loaded with {len(dataset.data)} subjects")
    for sub in dataset.data:
        print(f"  Subject {sub}: {len(dataset.data[sub])} sequences")
    gen = dataset.iter_generator()
    for traj, traj_multi in gen:
        print(f"Traj shape: {traj.shape}")
        break
