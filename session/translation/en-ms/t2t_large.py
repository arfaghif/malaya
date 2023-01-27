import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor import problems
import tensorflow as tf
import os
import logging

logger = logging.getLogger()
@@#logging.set_verbosity(tf.compat.v1.logging.DEBUG)

TRAIN_DATASETS = [
    [
        'https://f000.backblazeb2.com/file/malay-dataset/train-en-ms.tar.gz',
        ('train-en/left.txt', 'train-en/right.txt'),
    ]
]

TEST_DATASETS = [
    [
        'https://f000.backblazeb2.com/file/malay-dataset/test-en-ms.tar.gz',
        ('test-en/left.txt', 'test-en/right.txt'),
    ]
]


@registry.register_problem
class TRANSLATION32k(translate.TranslateProblem):
    @property
    def additional_training_datasets(self):
        """Allow subclasses to add training datasets."""
        return []

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_datasets = TRAIN_DATASETS + self.additional_training_datasets
        return train_datasets if train else TEST_DATASETS


os.system('mkdir t2t/train-large')
DATA_DIR = os.path.expanduser('t2t/data')
TMP_DIR = os.path.expanduser('t2t/tmp')
TRAIN_DIR = os.path.expanduser('t2t/train-large')
EXPORT_DIR = os.path.expanduser('t2t/export')
TRANSLATIONS_DIR = os.path.expanduser('t2t/translation')
EVENT_DIR = os.path.expanduser('t2t/event')
USR_DIR = os.path.expanduser('t2t/user')

PROBLEM = 'translatio_n32k'
t2t_problem = problems.problem(PROBLEM)

train_steps = 500000
eval_steps = 10
batch_size = 4096
save_checkpoints_steps = 25000
ALPHA = 0.1
schedule = 'continuous_train_and_eval'
MODEL = 'transformer'
HPARAMS = 'transformer_big'

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems

hparams = create_hparams(HPARAMS)
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA

RUN_CONFIG = create_run_config(
    model_dir = TRAIN_DIR,
    model_name = MODEL,
    save_checkpoints_steps = save_checkpoints_steps,
    num_gpus = 3,
)

tensorflow_exp_fn = create_experiment(
    run_config = RUN_CONFIG,
    hparams = hparams,
    model_name = MODEL,
    problem_name = PROBLEM,
    data_dir = DATA_DIR,
    train_steps = train_steps,
    eval_steps = eval_steps,
    # use_xla=True # For acceleration
)

tensorflow_exp_fn.train_and_evaluate()
