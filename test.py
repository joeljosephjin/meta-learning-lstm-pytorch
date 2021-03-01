import torch
# import learn2learn as l2l


class MetaLSTM():
	def __init__(self):
		pass

	def adapt(self, learner, data):
		pass

	def step(self):
		pass



data = torch.tensor([3, 4]), torch.tensor([7])

learner = torch.nn.Linear(2, 1)

MetaLearner = MetaLSTM()

MetaLearner.adapt(learner, data)

MetaLearner.step()