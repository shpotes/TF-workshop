import pathlib
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import preprocessing
from typing import List, Tuple


type_map = {'why': 0, 'who': 1, 'what': 2,
            'how': 3, 'where': 4, 'when': 5}

def build_source_from_metadata(metadata: pd.DataFrame,
                               data_dir: str,
                               mode: str) -> List[Tuple[str]]:
    df = metadata.copy().sample(frac=1).reset_index(drop=True)
    df = df[df['split'] == mode]
    df['filepath'] = df['filaname'].apply(
        lambda x: str((pathlib.Path(data_dir) / 'images' / x).resolve())
    )
    df['type'] = df['type'].apply(lambda x: type_map[x])
    source = list(zip(df['filepath'], *df.iloc[:, 2:-1].T.values))
    return source
    

def tokenize(raw_text, tok, **kwargs):
    tok_text = tok.texts_to_sequences(raw_text)
    pad_text = preprocessing.sequence.pad_sequences(tok_text,
                                                    padding='post',
                                                    **kwargs)
    return pad_text


def load(raw):
    # load images
    filepath = raw['image']
    image = tf.io.read_file(filepath)
    raw['image'] = tf.io.decode_jpeg(image)

    raw['correct_answer'] = 3
    return raw


def preprocess(raw):
    image = tf.image.resize(raw['image'], (224, 224))
    image = image / 127.5
    image = image - 1

    question_type = tf.one_hot(raw['question_type'], 6)
    correct_answer = tf.one_hot(raw['correct_answer'], 4)
    
    mask = tf.random.shuffle(tf.range(4))

    correct_answer = tf.gather(correct_answer, mask)
    answers = tf.transpose(tf.gather(tf.transpose(raw['answers']), mask))

    return (image, raw['question'], answers), (correct_answer, question_type)


def make_dataset(source, 
                 tokenizer,
                 training=False,
                 batch_size=1,
                 num_epochs=1,
                 num_parallel_calls=1,
                 shuffle_buffer_size=None):
    
    if not shuffle_buffer_size:
        shuffle_buffer_size = batch_size * 4
        
    img, que, ans1, ans2, ans3, ans4, q_type = zip(*source)

    # tokenize data
    ans = [tokenize(a, tokenizer, maxlen=20) for a in [ans1, ans2, ans3, ans4]]
    ans = np.stack(ans, axis=-1)
    que = tokenize(que, tokenizer)

    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(img),
        'question': list(que),
        'answers': list(ans),
        'question_type': list(q_type)
    })
    
    if training:
        ds.shuffle(shuffle_buffer_size)

    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(preprocess)
    ds = ds.repeat(count=num_epochs).batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds
