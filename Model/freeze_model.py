import tensorflow as tf

with tf.Session() as session:
    my_saver = tf.train.import_meta_graph('xor.tflearn.meta')
    my_saver.restore(session, tf.train.latest_checkpoint('.'))   

    frozen_graph = tf.graph_util.convert_variables_to_constants(
        session,
        session.graph_def,
        ['my_output/Sigmoid']
    )

    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph.SerializeToString())
