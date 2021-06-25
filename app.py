import flask
import psycopg2 
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import json


from psycopg2 import Error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importing modules required for BERT model
from official import nlp
from official.nlp import bert
# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

# Use pickle to load in the pre-trained model.
export_dir='./saved_model'
model = tf.saved_model.load(export_dir)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        input_string = flask.request.form['Input String']
        ### Connecting to an existing database to get the dataset
        try:
            # Connect to an ENKI database
            enki_db = psycopg2.connect("dbname=enki user=tsdbadmin host=latitude-enki-oasistech-69b2.a.timescaledb.io port=18043 password=xsluda83gk632stx")
            #"Enter the name of the task you want to perform:
            input_taskname = "Action Type Detector"
            mycursor = enki_db.cursor();
            sql = "select id from fewshot_task where name = '{0}';".format(input_taskname)
            # Executing query
            mycursor.execute(sql)
            task_id = mycursor.fetchone()

            # Reading the specified data from the Enki database for the above mentioned task
            df = pd.read_sql("select inputs, outputs from fewshot_data where fewshot_task_id = {0};".format(int(task_id[0])), enki_db)
            pd.set_option('display.expand_frame_repr', False);
            df = df.applymap(lambda x: (x)[0])

        except (Exception, Error) as error:
            print("Error while connecting to PostgreSQL", error)
        finally:
            if (enki_db):
                mycursor.close()
                enki_db.close()
        # Reading the class labels
        class_labels = list(df.outputs.unique())
        dict_classlabels = {class_label: categorical_label for categorical_label, class_label in enumerate(class_labels)}
        gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
        tf.io.gfile.listdir(gs_folder_bert)
        def tfrecord_creation(dataframe):
            text = dataframe.loc[:,'inputs']
            label = dataframe.loc[:,'outputs']

            # String text input feature
            text_feature = text.values
            # Class label features 
            label_feature = label.values
            
            #Creates a Dataset whose elements are slices of the given tensors
            text_features_dataset = tf.data.Dataset.from_tensor_slices((text_feature))
            label_features_dataset = tf.data.Dataset.from_tensor_slices((label_feature))
            data_string = [textfeature for textfeature in text_features_dataset]
            data_label = [labelfeature for labelfeature in label_features_dataset]
            datalabel = tf.stack(data_label,0)


            #The following code rebuilds the tokenizer that was used by the base Bert model:
            
            # Set up tokenizer to generate Tensorflow dataset
            tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),do_lower_case=True)

            #Encode the sentences
            #This input is expected to start with a [CLS] "This is a classification problem" token, and each sentence should end with a [SEP] "Separator" token.
            
            #Start by encoding all the sentences while appending a [SEP] token, and packing them into ragged-tensors:
            def append_sep_token(s):
                tokens = list(tokenizer.tokenize(s.numpy()))
                tokens.append('[SEP]')
                return tokenizer.convert_tokens_to_ids(tokens)

            def apply_tokens(datastring, tokenizer):
                sentence = tf.ragged.constant([append_sep_token(s) for s in datastring])
                cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence.shape[0]
                input_word_ids = tf.concat([cls, sentence], axis=-1)
                input_mask = tf.ones_like(input_word_ids).to_tensor()
                type_cls = tf.zeros_like(cls)
                type_s1 = tf.zeros_like(sentence)
                input_type_ids = tf.concat([type_cls, type_s1], axis=-1).to_tensor()
                inputs = {
                    'input_word_ids': input_word_ids.to_tensor(),
                    'input_mask': input_mask,
                    'input_type_ids': input_type_ids}
                return inputs
            
            tokenized_data = apply_tokens(data_string, tokenizer)

            return (tokenized_data, datalabel)
        

        idf_test= pd.DataFrame(data=[[input_string.encode('utf-8'),len(class_labels)+1]],columns = ['inputs', 'outputs'])
        tf_test_string, tf_random_label = tfrecord_creation(df_test)
        reloaded_result_string = model([tf_test_string['input_word_ids'],
                                    tf_test_string['input_mask'],
                                    tf_test_string['input_type_ids']], training=False)
        predicted_label_encoded_string = tf.argmax(reloaded_result_string, axis=1).numpy()
        reversed_dictionary = dict(map(reversed, dict_classlabels.items()))
        prediction = [reversed_dictionary[i] for i in predicted_label_encoded_string][0])]
        
        return flask.render_template('main.html',
                                     original_input={'Input String':input_string},
                                     result=prediction,
                                     )
