__author__ = 'manabchetia'

import graphlab as gl

# Load the data
train = gl.SFrame.read_csv('../data/train.csv')
test = gl.SFrame.read_csv('../data/test.csv')
sample = gl.SFrame.read_csv('../data/sampleSubmission.csv')

del train['id']

# Train a model
model = gl.boosted_trees_classifier.create(train, target='target', max_iterations=100)

# Make submission
preds = model.predict_topk(test, output_type='probability', k=9)
preds['id'] = preds['id'].astype(int) + 1
preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
preds = preds.sort('id')

preds_sf = gl.SFrame(preds)
preds_sf.save('../output/graphlab.csv', format='csv')


