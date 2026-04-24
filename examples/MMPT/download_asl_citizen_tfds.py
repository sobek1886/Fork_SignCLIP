import tensorflow_datasets as tfds

tfds.load(
    'asl_citizen',
    data_dir='/scratch-shared/psobecki/tensorflow_datasets',
    download=True,
)
print("Done.")
