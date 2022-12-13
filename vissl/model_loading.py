import torch 

model = torch.load('/home/ubuntu/byol/improved-net-2.pt')

print(model.keys())

# torch.save(model['classy_state_dict']['base_model']['model']['trunk'], 'model_epoch36.torch')

# print(model['classy_state_dict']['base_model']['model'])
# print(model['classy_state_dict']['base_model']['model']['trunk'].keys())