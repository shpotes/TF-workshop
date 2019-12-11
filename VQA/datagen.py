import pathlib
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Tuple


type_map = {'why': 0, 'who': 1, 'what': 2,
            'how': 3, 'where': 4, 'when': 5}

def build_source_from_metadata(metadata: pd.DataFrame,
                               data_dir: str,
                               mode: str) -> List[Tuple[str]]:
    df = metadata.copy().sample(frac=1).reset_index(drop=True)
    df = df[df['split'] == mode]
    df['filepath'] = df['filaname'].apply(
        lambda x: str((pathlib.Path(data_dir) / x).resolve())
    )
    df['type'] = df['type'].apply(lambda x: type_map[x])
    source = list(zip(df['filepath'], *df.iloc[:, 2:-1].T.values))
    return source
    
    
def load(raw, type_map):
    filepath = raw['image']
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img)
    return img, raw['ans_text'], [0, 0, 0, 1], raw['type']


def propreocess(img, ans, out, typ, tok):
    img = tf.keras.applications.vgg16.preprocess_input(img)
    # TODO preprocess text


def make_dataset(source, 
                 preprocess,
                 training=False,
                 batch_size=1,
                 num_epochs=1,
                 num_parallel_calls=1,
                 shuffle_buffer_size=None):
    
    if not shuffle_buffer_size:
        shuffle_buffer_size = batch_size * 4
        
    image, *ans, q_type = zip(*source)
    
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(image),
        'ans_text': list(ans),
        'type': list(q_type)
    })
    
    if training:
        ds.shuffle(shuffle_buffer_size)
    
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x, y: prepocess(x, y))
    
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)
    
    return ds
