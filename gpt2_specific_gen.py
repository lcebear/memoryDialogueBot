#!/usr/bin/env python3

#import fire
import re
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


import model, sample, encoder


model_name='run_18_pc_history'
seed=None
nsamples=8
batch_size=8
length=40
temperature=1
top_k=0
top_p=0.9
threads = 8

models_dir = os.path.dirname(os.path.realpath(__file__)) + r"\checkpoint"
#models_dir = os.path.expanduser(os.path.expandvars(models_dir))

#start tf session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
if threads > 0:
    config.intra_op_parallelism_threads = threads
    config.inter_op_parallelism_threads = threads

gpt_session = tf.compat.v1.Session(config=config)
global graph
graph = tf.compat.v1.get_default_graph()


with graph.as_default():
    #load some stuff
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context)


    saver = tf.compat.v1.train.Saver(allow_empty=True)
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    gpt_session.run(tf.compat.v1.global_variables_initializer())
    print('Loading pretrained model', ckpt)
    saver.restore(gpt_session, ckpt)
    
    
#--------------------
    #setting context (set_context didn't work)
    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    np.random.seed(seed)
    tf.set_random_seed(seed)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
        )
    
#---------------------------------------------------------------
def generate(prefix=None,
             sess=gpt_session,
             return_as_list=True,
             truncate="<|endoftext|>",
             sample_delim='=' * 20 + '\n',
             seed=None,
             nsamples=8,
             batch_size=8,
             length=40,
             temperature=1.0,
             top_k=0,
             top_p=0.9,
             include_prefix=False,
             enc=enc,
             hparams=hparams
             ):


    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None
        
    global graph
    with graph.as_default():
    
        if prefix:
        #    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
            context_tokens = enc.encode(prefix)
        #makes random predictable 
        #np.random.seed(seed)
        #makes random predictable
        #tf.compat.v1.set_random_seed(seed)
        #tf.compat.v1.random.set_random_seed(seed) #I think this is the correct code

        #output = sample.sample_sequence(
        #    hparams=hparams,
        #    length=min(length, 1023 - (len(context_tokens) if prefix else 0)),
        #    start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        #    context=context if prefix else None,
        #    batch_size=batch_size,
        #    temperature=temperature, top_k=top_k, top_p=top_p
        #)[:, 1:]

        generated = 0
        gen_texts = []
        while generated < nsamples:
            if not prefix:
                out = sess.run(output)
            else:
                out = sess.run(output, feed_dict={
                        context: batch_size * [context_tokens]
                    })
            for i in range(batch_size):
                generated += 1
                gen_text = enc.decode(out[i])
                if prefix:
                    gen_text = enc.decode(context_tokens[:1]) + gen_text
                if truncate:
                    truncate_esc = re.escape(truncate)
                    if prefix and not include_prefix:
                        prefix_esc = re.escape(prefix)
                        pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                             truncate_esc)
                    else:
                        pattern = '(.*?)(?:{})'.format(truncate_esc)

                    trunc_text = re.search(pattern, gen_text, re.S)
                    if trunc_text:
                        gen_text = trunc_text.group(1)
                gen_text = gen_text.lstrip('\n')

                if not return_as_list:
                    print("{}\n{}".format(gen_text, sample_delim), end='')
                gen_texts.append(gen_text)

        if return_as_list:
            return gen_texts

def set_context(
             seed=None,
             batch_size=8,
             length=40,
             temperature=1.0,
             top_k=0,
             top_p=0.9,
             hparams=hparams):

    global graph
    with graph.as_default():
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

#set_context()



#-------------------------------------------------------------


#while(True):
#    text = input('> ')
#    generate(prefix=text)
