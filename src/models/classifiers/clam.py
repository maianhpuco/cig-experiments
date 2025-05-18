# class ClamLogitWrapper(nn.Module):
#     def __init__(self, clam_model):
#         super().__init__()
#         self.clam = clam_model

#     def forward(self, x):
#         logits, *_ = self.clam(x, attention_only=False)
#         return logits

# wrapped_model = ClamLogitWrapper(clam_model)
# logits = wrapped_model(inputs)
 