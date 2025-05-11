import pandas as pd

# todo
# use prompts, llms to analyse SAE features (Wd)
# methodology: appendix C in https://arxiv.org/pdf/2408.00657

INTERPRETER = """\
You are a meticulous {type} researcher conducting an important investigation into a certain
neuron in a language model trained on {subject} papers. Your task is to figure out what
sort of behaviour this neuron is responsible for – namely, on what general concepts, features,
themes, methodologies or topics does this neuron fire? Here’s how you’ll complete the task:

# INPUT DESCRIPTION:
You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.
1. You will be given several examples of text that activate the neuron, along with a
number being how much it was activated. This means there is some feature, theme,
methodology, topic or concept in this text that ‘excites’ this neuron.
2. You will also be given several examples of text that don’t activate the neuron. This
means the feature, topic or concept is not present in these texts.

# OUTPUT DESCRIPTION:
Given the inputs provided, complete the following tasks.
1. Based on the MAX ACTIVATING EXAMPLES provided, write down potential topics,
concepts, themes, methodologies and features that they share in common. These
will need to be specific - remember, all of the text comes from subject, so these
need to be highly specific subject concepts. You may need to look at different
levels of granularity (i.e. subsets of a more general topic). List as many as you can
think of. Give higher weight to concepts more present/prominent in examples with
higher activations.
2. Based on the zero activating examples, rule out any of the topics/concepts/features
listed above that are in the zero-activating examples. Systematically go through your
list above.
3. Based on the above two steps, perform a thorough analysis of which feature, concept
or topic, at what level of granularity, is likely to activate this neuron. Use Occam’s
razor, as long as it fits the provided evidence. Be highly rational and analytical here.
4. Based on step 4, summarise this concept in 1-8 words, in the form FINAL:
{explanation}. Do NOT return anything after these 1-8 words.

Here are the max-activating examples:
{max_activating_examples}

Here are the zero-activating examples:
{zero_activating_examples}

Work through the steps thoroughly and analytically to interpret our neuron."""

PREDICTOR = """\
You are a {subject} expert that is predicting which abstracts will activate a certain neuron
in a language model trained on {subject} papers. Your task is to predict which of the
following abstracts will activate the neuron the most. Here’s how you’ll complete the task:

# INPUT DESCRIPTION:
You will be given the description of the type of paper abstracts on
which the neuron activates. This description will be short. You will then be given an abstract.
Based on the concept of the abstract, you will predict whether the neuron will activate or not.

# OUTPUT DESCRIPTION:
Given the inputs provided, complete the following tasks.
1. Based on the description of the type of paper abstracts on which the neuron activates,
reason step by step about whether the neuron will activate on this abstract or not.
Be highly rational and analytical here. The abstract may not be clear cut - it may
contain topics/concepts close to the neuron description, but not exact. In this case,
reason thoroughly and use your best judgement. However, do not speculate on topics
that are not present in the abstract.
2. Based on the above step, predict whether the neuron will activate on this abstract
or not. If you predict it will activate, give a confidence score from 0 to 1 (i.e. 1
if you’re certain it will activate because it contains topics/concepts that match the
description exactly, 0 if you’re highly uncertain). If you predict it will not activate,
give a confidence score from -1 to 0.
3. Provide the final confidence score in the form PREDICTION: (your prediction)
e.g. PREDICTION: 0.5. Do NOT return anything after this.

Here is the description/interpretation of the type of paper abstracts on which the neuron
activates: {description}

Here is the abstract to predict: {abstract}

Work through the steps thoroughly and analytically to predict whether the neuron will activate
on this abstract."""


def analyze_features(model):
    dictionary = pd.DataFrame(model.decoder.weight.T.cpu().detach().numpy())
