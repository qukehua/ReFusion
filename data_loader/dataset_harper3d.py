import os
from glob import glob
import pickle as pkl
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


class DatasetHarper3D(Dataset):
    """
    Data loader for the Harper (3D) dataset, compatible with TransFusion.
    Human: 21 joints, Spot robot: 23 joints, total: 44 joints
    """

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 data_path='./data/harper3d', include_spot=True, fps='30hz',
                 use_data_aug=False, aug_rotate_prob=0.5, aug_reverse_prob=0.3):
        """
        Args:
            mode: 'train' or 'test'
            t_his: history frames (observation)
            t_pred: prediction frames
            actions: 'all' or list of action names
            use_vel: whether to use velocity
            data_path: path to harper3d dataset root
            include_spot: whether to include spot robot joints
            fps: frame rate version '30hz' or '120hz'
        """
        self.use_vel = use_vel
        self.data_path = os.path.join(data_path, fps)  # e.g., ./data/harper3d/30hz
        self.include_spot = include_spot
        self.actions_filter = actions
        self.fps = fps
        self.use_data_aug = use_data_aug and mode == 'train'
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_reverse_prob = aug_reverse_prob
        
        super().__init__(mode, t_his, t_pred, actions)
        
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        """
        Prepare Harper3D data in TransFusion format.
        Data structure: self.data[subject][action] = sequence (frames, joints, 3)
        """
        # HARPER official human links:
        # https://github.com/intelligolabs/HARPER/blob/main/tools/links.py
        human_links = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (3, 5), (5, 6), (6, 7), (7, 8),
            (3, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (6, 13), (10, 17),
        ]
        # Tree parents are still kept for compatibility; visualization uses
        # official links above when available.
        human_parents = [-1, 0, 1, 2, 3, 3, 5, 6, 7, 3, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        
        if self.include_spot:
            # HARPER official Spot links. Spot joints are shifted by 21 because
            # they are stored after the human joints.
            spot_links = [
                (4, 10), (7, 1),
                (1, 2), (2, 3),
                (4, 5), (5, 6),
                (7, 8), (8, 9),
                (10, 11), (11, 12),
                (13, 14), (14, 16), (16, 15), (15, 13),
                (17, 18), (18, 20), (20, 19), (19, 17),
                (13, 17), (14, 18), (15, 19), (16, 20),
                (0, 21), (21, 22),
            ]
            spot_parents = [-1] * 23
            for parent, child in spot_links:
                if spot_parents[child] == -1:
                    spot_parents[child] = parent
            all_parents = human_parents + [p + 21 if p >= 0 else -1 for p in spot_parents]
            all_links = human_links + [(a + 21, b + 21) for a, b in spot_links]
            self.num_human_joints = 21
            self.num_spot_joints = 23
            self.total_joints = 44
            # Left/right groups for coloring in visualization.
            joints_left = [1, 2, 3, 4, 13, 14, 15, 16]
            joints_right = [5, 6, 7, 8, 17, 18, 19, 20]
        else:
            all_parents = human_parents
            all_links = human_links
            self.num_human_joints = 21
            self.num_spot_joints = 0
            self.total_joints = 21
            joints_left = [1, 2, 3, 4, 13, 14, 15, 16]
            joints_right = [5, 6, 7, 8, 17, 18, 19, 20]
        
        self.skeleton = Skeleton(
            parents=all_parents,
            joints_left=joints_left,
            joints_right=joints_right,
            links=all_links
        )
        
        # All joints are kept (no removal)
        self.kept_joints = np.arange(self.total_joints)
        self.removed_joints = set()
        
        # Define subjects
        self.subjects_split = {
            'train': ['avo', 'bn', 'cun', 'el', 'h', 'jk', 'j', 'mt', 'ric', 'ry', 'sh', 'son'],
            'test': ['t', 'toa', 'xu', 'xy', 'yf']
        }
        self.subjects = self.subjects_split[self.mode]
        
        # Action list
        self.all_actions = [
            'act1_0', 'act1_45', 'act1_90', 'act1_180',
            'act2', 'act3', 'act4', 'act5', 'act6', 'act7',
            'act8', 'act9', 'act10', 'act11', 'act12'
        ]
        
        self.process_data()

    def process_data(self):
        """
        Load and process Harper3D pkl files.
        """
        data_folder = os.path.join(self.data_path, self.mode)
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"Data folder {data_folder} not found. "
                f"Please download Harper3D dataset and place it in {self.data_path}"
            )
        
        pkl_files = glob(os.path.join(data_folder, "*.pkl"))
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No pkl files found in {data_folder}")
        
        self.data = {}
        
        for pkl_file in pkl_files:
            sequence_dict = load_pkl(pkl_file)
            
            # Get subject and action from first frame
            first_frame = sequence_dict[0]
            subject = first_frame['subject']
            action = first_frame['action']
            
            # Filter by subject
            if subject not in self.subjects:
                continue
            
            # Filter by action
            if self.actions_filter != 'all':
                if isinstance(self.actions_filter, str):
                    if action != self.actions_filter:
                        continue
                elif action not in self.actions_filter:
                    continue
            
            # Extract sequence data
            frames = sorted(sequence_dict.keys())
            human_seq = np.array([sequence_dict[f]['human_joints_3d'] for f in frames])  # (T, 21, 3)
            
            if self.include_spot:
                spot_seq = np.array([sequence_dict[f]['spot_joints_3d'] for f in frames])  # (T, 23, 3)
                seq = np.concatenate([human_seq, spot_seq], axis=1)  # (T, 44, 3)
            else:
                seq = human_seq  # (T, 21, 3)
            
            # Process: make relative to root joint (joint 0)
            if self.use_vel:
                v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                v = np.append(v, v[[-1]], axis=0)
            
            # Make positions relative to root (keep root at origin)
            seq[:, 1:] -= seq[:, :1]
            
            if self.use_vel:
                seq = np.concatenate((seq, v), axis=1)
            
            # Store in data dict
            if subject not in self.data:
                self.data[subject] = {}
            
            # Use filename as unique action key if action already exists
            action_key = action
            counter = 1
            while action_key in self.data[subject]:
                action_key = f"{action}_{counter}"
                counter += 1
            
            self.data[subject][action_key] = seq

    def _apply_scene_rotation(self, sample):
        """Rotate the whole scene around the vertical z-axis."""
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ], dtype=sample.dtype)
        return np.matmul(sample, rot.T)

    def _apply_sequence_reverse(self, sample):
        """
        Reverse the temporal order and swap history/future semantics.

        Following the paper's idea, after reversal we keep the last t_his frames
        of the original sequence as the observation by taking the reversed clip.
        """
        return sample[:, ::-1].copy()

    def augment_sample(self, sample):
        if np.random.uniform() < self.aug_rotate_prob:
            sample = self._apply_scene_rotation(sample)
        if np.random.uniform() < self.aug_reverse_prob:
            sample = self._apply_sequence_reverse(sample)
        return sample

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for _ in range(num_samples // batch_size):
            sample = []
            for _ in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug and self.use_data_aug:
                sample = self.augment_sample(sample)
            yield sample

    def get_sample_with_action(self, action_name):
        """Sample a sequence from a specific action."""
        available_subjects = [s for s in self.subjects if s in self.data]
        for subject in available_subjects:
            for act_key in self.data[subject].keys():
                if action_name in act_key:
                    seq = self.data[subject][act_key]
                    if seq.shape[0] >= self.t_total:
                        fr_start = np.random.randint(seq.shape[0] - self.t_total)
                        return seq[None, fr_start:fr_start + self.t_total]
        return None


def gen_velocity(m):
    dm = np.zeros_like(m)
    dm[:, 1:] = m[:, 1:] - m[:, :-1]
    dm[:, 0] = dm[:, 1]
    return dm


if __name__ == '__main__':
    np.random.seed(0)
    dataset = DatasetHarper3D('train', t_his=25, t_pred=100, data_path='./data/harper3d')
    print(f"Dataset loaded with {len(dataset.data)} subjects")
    for sub in dataset.data:
        print(f"  Subject {sub}: {len(dataset.data[sub])} sequences")
    generator = dataset.sampling_generator()
    for data in generator:
        print(f"Sample shape: {data.shape}")
        break

