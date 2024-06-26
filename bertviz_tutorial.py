!pip install bertviz

# Load model and retrieve attention weights

from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel

model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)

"""# Head View
<b>The head view visualizes attention in one or more heads from a single Transformer layer.</b> Each line shows the attention from one token (left) to another (right). Line weight reflects the attention value (ranges from 0 to 1), while line color identifies the attention head. When multiple heads are selected (indicated by the colored tiles at the top), the corresponding  visualizations are overlaid onto one another.  For a more detailed explanation of attention in Transformer models, please refer to the [blog](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1).

## Usage
👉 **Hover** over any **token** on the left/right side of the visualization to filter attention from/to that token. <br/>
👉 **Double-click** on any of the **colored tiles** at the top to filter to the corresponding attention head.<br/>
👉 **Single-click** on any of the **colored tiles** to toggle selection of the corresponding attention head. <br/>
👉 **Click** on the **Layer** drop-down to change the model layer (zero-indexed).

"""

head_view(attention, tokens, sentence_b_start)

"""# Model View
<b>The model view provides a birds-eye view of attention throughout the entire model</b>. Each cell shows the attention weights for a particular head, indexed by layer (row) and head (column).  The lines in each cell represent the attention from one token (left) to another (right), with line weight proportional to the attention value (ranges from 0 to 1).  For a more detailed explanation, please refer to the [blog](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1).

## Usage
👉 **Click** on any **cell** for a detailed view of attention for the associated attention head (or to unselect that cell). <br/>
👉 Then **hover** over any **token** on the left side of detail view to filter the attention from that token.
"""

model_view(attention, tokens, sentence_b_start)

"""# Neuron View
<b>The neuron view visualizes the intermediate representations (e.g. query and key vectors) that are used to compute attention.</b> In the collapsed view (initial state), the lines show the attention from each token (left) to every other token (right). In the expanded view, the tool traces the chain of computations that produce these attention weights. For a detailed explanation of the attention mechanism, please refer to the [blog](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1).

## Usage
👉 **Hover** over any of the tokens on the left side of the visualization to filter attention from that token.<br/>
👉 Then **click** on the **plus** icon that is revealed when hovering. This exposes the query vectors, key vectors, and other intermediate representations used to compute the attention weights. Each color band represents a single neuron value, where color intensity indicates the magnitude and hue the sign (blue=positive, orange=negative).<br/>
👉 Once in the expanded view, **hover** over any other **token** on the left to see the associated attention computations.<br/>
👉 **Click** on the **Layer** or **Head** drop-downs to change the model layer or head (zero-indexed).

"""

from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

model_type = 'bert'
model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
show(model, model_type, tokenizer, sentence_a, sentence_b, layer=4, head=3)

"""# Next steps

Visit the [github repository](https://github.com/jessevig/bertviz) for full documentation and additional use cases. Check out the [blog](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1) for a deep dive on BertViz and the attention mechanism.
"""