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
                 data_path='./data/harper3d', include_spot=True, fps='30hz'):
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
        
        super().__init__(mode, t_his, t_pred, actions)
        
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        """
        Prepare Harper3D data in TransFusion format.
        Data structure: self.data[subject][action] = sequence (frames, joints, 3)
        """
        # Harper3D skeleton definition for human (21 joints)
        # Parent indices for 21 human joints (simplified kinematic chain)
        human_parents = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 8, 15, 16, 0, 18, 19]
        
        if self.include_spot:
            # Spot robot has 23 joints, add them after human joints
            # Spot is independent, so root parent is -1 (shifted by 21)
            spot_parents = [-1] + [i + 21 for i in range(22)]
            all_parents = human_parents + [p + 21 if p >= 0 else -1 for p in spot_parents]
            self.num_human_joints = 21
            self.num_spot_joints = 23
            self.total_joints = 44
            # For skeleton, we use simplified left/right
            joints_left = list(range(5, 8)) + list(range(12, 15))  # human left side
            joints_right = list(range(2, 5)) + list(range(15, 18))  # human right side
        else:
            all_parents = human_parents
            self.num_human_joints = 21
            self.num_spot_joints = 0
            self.total_joints = 21
            joints_left = [5, 6, 7, 12, 13, 14]
            joints_right = [2, 3, 4, 15, 16, 17]
        
        self.skeleton = Skeleton(
            parents=all_parents,
            joints_left=joints_left,
            joints_right=joints_right
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

