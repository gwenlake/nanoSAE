import pandas as pd

# todo
# use prompts, llms to analyse SAE features (Wd)
# methodology: appendix C in https://arxiv.org/pdf/2408.00657

def analyze_features(model):
    features = pd.DataFrame(model.decoder.weight.T.cpu().detach().numpy())
